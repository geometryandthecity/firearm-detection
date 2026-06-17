"""Headless multi-view renderer for the image classifier (two interchangeable backends).

Turns one ``.obj`` mesh into a fixed set of shaded views taken from cameras
spaced evenly around it, which is the input an MVCNN-style classifier consumes.
The mesh is recentered and scaled to the unit bounding sphere and photographed
by an orthographic camera swung around the azimuth at a fixed elevation, so the
framing is identical for every shape regardless of its original size/pose.

Two backends produce those views; pick one with the ``RENDER_BACKEND``
environment variable (default ``pyrender`` -- ~3.6x faster per shape than ``bpy``
on cv12 with detector AUC held; set ``RENDER_BACKEND=bpy`` for Blender):

* ``bpy`` -- Blender's Workbench engine (``BLENDER_WORKBENCH``) in single-colour
  solid mode under the built-in STUDIO light, driven through the ``bpy`` *pip
  module* with no GUI/display. Shading is ~0.01s/view, but each shape pays
  Blender's scene reset + OBJ import, which dominate wall time on high-poly
  meshes over NFS.
* ``pyrender`` -- an offscreen GPU rasterizer on EGL (no display). The mesh is
  read with gpytoolbox, sanitized, and drawn with a grey matte material under a
  camera-relative key+fill light rig (a flat headlamp was measured to cost
  detector AUC); the GL context and renderer are created once per process and
  reused across every shape/view. This skips Blender's per-shape scene reset and
  importer entirely, which is where the time went.
"""
import contextlib
import os
import sys
from math import cos, sin, pi, radians

# Which backend renders the views: "bpy" (Blender/Workbench) or "pyrender"
# (offscreen EGL GPU rasterizer). Read once at import so a single evaluate.py
# process is internally consistent; A/B by setting RENDER_BACKEND.
RENDER_BACKEND = os.environ.get("RENDER_BACKEND", "pyrender").strip().lower()

# Bump the per-backend tag whenever that backend's rendered *appearance* changes
# (engine, lighting, material, background) so the detector's render cache
# invalidates instead of silently reusing visually-stale PNGs. The backend name
# is part of the tag so cached PNGs from two engines never mix.
_STYLE_BY_BACKEND = {
    "bpy": "workbench-v1",          # Workbench solid / STUDIO-light look
    "pyrender": "pyrender-egl-v2",  # EGL grey matte under a raking key+fill rig
}
RENDER_STYLE = _STYLE_BY_BACKEND.get(RENDER_BACKEND, f"{RENDER_BACKEND}-v1")


@contextlib.contextmanager
def _quiet():
    """Silence Blender's render/shader-compile chatter on stdout+stderr.

    Blender logs ("Saved: ...", "OBJ import took ...", render progress) at the
    C level, so redirecting Python's ``sys.stdout`` is not enough -- the
    file descriptors themselves (1 and 2) are pointed at ``/dev/null`` for the
    duration of a render. C stdio also *buffers* in userspace, so we flush libc's
    streams (``fflush(NULL)``) while the fds still point at /dev/null; otherwise
    buffered lines leak to the restored terminal at process exit. The method's
    own progress logging happens outside this block, so it stays visible.
    """
    try:
        import ctypes
        libc = ctypes.CDLL(None)
    except Exception:                       # pragma: no cover - platform fallback
        libc = None
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out, old_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        if libc is not None:
            try:
                libc.fflush(None)           # flush C stdio while fds -> /dev/null
            except Exception:               # pragma: no cover
                pass
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(devnull)
        os.close(old_out)
        os.close(old_err)


def _import_and_normalize(bpy, Vector, obj_path):
    """Import ``obj_path``, join its meshes, and recenter + unit-scale the result.

    Returns the single joined mesh object placed so its bounding box is centered
    at the origin and fits inside the unit sphere, which makes the orthographic
    framing identical for every shape regardless of its original size/position.
    Raises ``RuntimeError`` if the file contains no mesh (degenerate input the
    caller scores as benign).
    """
    bpy.ops.wm.obj_import(filepath=obj_path)
    meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {obj_path}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # bound_box gives the 8 local-space corners; after transform_apply the
    # object's matrix is identity so local == world. Center on the bbox midpoint
    # and scale by the bounding-sphere radius so the whole shape fits in unit r.
    mw = obj.matrix_world
    corners = [mw @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector((0.0, 0.0, 0.0))) / 8.0
    radius = max((c - center).length for c in corners)
    s = 1.0 / radius if radius > 0 else 1.0
    obj.scale = (s, s, s)
    obj.location = (-center.x * s, -center.y * s, -center.z * s)
    bpy.context.view_layer.update()
    return obj


def _build_scene(bpy, img_size, ortho_scale):
    """Reset to an empty file and wire up the camera, Workbench shading, engine.

    Returns ``(scene, cam_obj)``. Workbench derives the whole look from
    ``scene.display.shading`` (one grey colour under the built-in STUDIO light)
    and ignores per-object materials and world lighting, so no world/sun/material
    is created. The camera is orthographic and aimed at the origin via a TRACK_TO
    constraint, so callers only move its *location* around the azimuth and the
    framing/aim takes care of itself.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Empty at the origin for the camera to track.
    target = bpy.data.objects.new("ic_target", None)
    scene.collection.objects.link(target)
    target.location = (0.0, 0.0, 0.0)

    cam_data = bpy.data.cameras.new("ic_cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam_obj = bpy.data.objects.new("ic_cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    con = cam_obj.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = img_size
    scene.render.resolution_y = img_size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    # Workbench ignores node materials/world lighting -- drive the look via the
    # solid-mode display shading: one neutral grey colour lit by the built-in
    # STUDIO light, with outline/cavity/shadows off for a clean matte. Property
    # names are guarded in case they shift across bpy versions.
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.8, 0.8, 0.8)
    for prop, val in (("show_object_outline", False), ("show_cavity", False),
                      ("show_shadows", False), ("background_type", "VIEWPORT")):
        try:
            setattr(sh, prop, val)
        except Exception:
            pass
    try:
        sh.background_color = (0.5, 0.5, 0.5)
    except Exception:
        pass
    # Render-time anti-aliasing samples (16 is ample for a small classifier input).
    try:
        scene.display.render_aa = "16"
    except Exception:
        pass

    return scene, cam_obj


def _eye_at(azimuth_deg, elevation_deg, dist):
    """A single point on the viewing sphere at ``(azimuth, elevation)`` (Z-up)."""
    az = radians(azimuth_deg)
    el = radians(elevation_deg)
    return (dist * cos(el) * cos(az), dist * cos(el) * sin(az), dist * sin(el))


def _camera_eyes(n_views, elevation_deg, dist):
    """The ``n_views`` camera positions, evenly spaced around the azimuth at a
    fixed elevation on a sphere of radius ``dist`` (Z-up). Shared by both backends
    so the two engines photograph the shape from the identical ring of viewpoints.
    """
    return [_eye_at(360.0 * i / n_views, elevation_deg, dist) for i in range(n_views)]


def _import_bpy():
    """Import and return ``bpy``, robust to ``multiprocessing`` *spawn*.

    The ``bpy`` pip package's top-level ``__init__`` is a launcher: it registers
    the ``_bpy`` C-extension and only then prepends its inner
    ``bpy/<ver>/scripts/modules`` dir to ``sys.path``. So once a parent process
    has imported ``bpy`` (e.g. to render the training set), that inner dir is on
    ``sys.path`` -- and ``spawn`` copies the parent's ``sys.path`` into the child.
    The per-shape eval-timeout worker (``base.py``) is exactly such a spawn child,
    so its ``import bpy`` resolves to the *inner* package, skips the launcher, and
    dies with ``ModuleNotFoundError: No module named '_bpy'`` (every render then
    fails and the shape is wrongly scored benign). Dropping the inherited inner
    entries and retrying lets the launcher run and bootstrap ``_bpy``. Only the
    failure path does any work, so a normal first import is untouched.
    """
    try:
        import bpy
        return bpy
    except ModuleNotFoundError:
        for p in [q for q in sys.path
                  if q.replace(os.sep, "/").rstrip("/").endswith("scripts/modules")]:
            sys.path.remove(p)
        sys.modules.pop("bpy", None)
        import bpy
        return bpy


def _render_views_bpy(obj_path, out_dir, n_views, img_size,
                      elevation_deg, dist, ortho_scale):
    """Blender/Workbench backend (see module docstring). ``bpy`` is imported here
    so merely importing this module never pulls in Blender."""
    bpy = _import_bpy()
    from mathutils import Vector

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with _quiet():
        scene, cam_obj = _build_scene(bpy, img_size, ortho_scale)
        _import_and_normalize(bpy, Vector, obj_path)  # adds the mesh to the scene

        for i, eye in enumerate(_camera_eyes(n_views, elevation_deg, dist)):
            cam_obj.location = eye
            bpy.context.view_layer.update()
            out = os.path.join(out_dir, f"view{i}.png")
            scene.render.filepath = out
            bpy.ops.render.render(write_still=True)
            paths.append(out)
    return paths


# --- pyrender (offscreen EGL) backend --------------------------------------
# pyrender selects its GL platform from PYOPENGL_PLATFORM at import time; pin it
# to EGL (headless, GPU) before the first import. A single OffscreenRenderer (and
# its GL context) is created lazily and reused for every shape/view in the
# process -- recreating it per shape is the bulk of pyrender's avoidable cost.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
_PYRENDER_CACHE = {}  # img_size -> (pyrender_module, OffscreenRenderer); module-level for reuse


def _look_at(eye, target, up):
    """4x4 camera-to-world for a camera at ``eye`` looking at ``target``.

    Follows glTF/pyrender convention: the camera looks down its local -Z with
    local +Y up, so this is the pose pyrender expects for both the camera and a
    co-located ("headlamp") directional light.
    """
    import numpy as np
    eye = np.asarray(eye, float)
    f = eye - np.asarray(target, float)            # camera +Z points away from target
    f /= (np.linalg.norm(f) or 1.0)
    r = np.cross(np.asarray(up, float), f)
    r /= (np.linalg.norm(r) or 1.0)
    u = np.cross(f, r)
    M = np.eye(4)
    M[:3, 0] = r
    M[:3, 1] = u
    M[:3, 2] = f
    M[:3, 3] = eye
    return M


def _pyrender_renderer(img_size, supersample):
    """Lazily build and cache the pyrender module + a reusable OffscreenRenderer
    sized for supersampled rendering (downsampled later for cheap anti-aliasing)."""
    key = img_size * supersample
    if key not in _PYRENDER_CACHE:
        import pyrender  # noqa: E402  (imported lazily; needs PYOPENGL_PLATFORM set)
        r = pyrender.OffscreenRenderer(viewport_width=key, viewport_height=key)
        _PYRENDER_CACHE[key] = (pyrender, r)
    return _PYRENDER_CACHE[key]


def _load_normalized_mesh(obj_path):
    """Read ``obj_path`` with gpytoolbox, sanitize away loader-exploit garbage
    (NaN / out-of-range / repeated-index faces, dangling verts), and recenter +
    unit-bounding-sphere scale the vertices -- matching the bpy path's framing.

    Returns ``(V, F)`` float/int arrays, or raises ``RuntimeError`` for a file
    with no usable triangles so the caller scores it as a degenerate (benign)
    input, exactly as the bpy backend does for a mesh-less import.
    """
    import numpy as np
    import gpytoolbox as gpy
    from ..utils import sanitize

    try:
        V, F = gpy.read_mesh(obj_path)
    except Exception as exc:
        raise RuntimeError(f"unreadable mesh {obj_path}: {exc!r}")
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        raise RuntimeError(f"no triangles in {obj_path}")
    V, F = sanitize(V, F)
    if len(F) == 0 or len(V) == 0:
        raise RuntimeError(f"no mesh survived sanitize for {obj_path}")
    lo = V.min(0)
    hi = V.max(0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))   # bbox bounding-sphere radius (matches bpy)
    s = 1.0 / radius if radius > 0 else 1.0
    return (V - center) * s, F


def _render_views_pyrender(obj_path, out_dir, n_views, img_size,
                           elevation_deg, dist, ortho_scale,
                           supersample=2, bg=0.5, base_color=0.85, ambient=0.32,
                           key_intensity=3.2, fill_intensity=1.0):
    """Offscreen EGL backend (see module docstring): grey matte mesh under a
    camera-relative key+fill light rig, orthographic camera ringed around the
    shape. A high raking key off to one side plus a soft fill from the other
    reveal 3D form (a co-located headlamp shades everything flat and was measured
    to cost detector AUC), while keeping the shading consistent across views.
    Renders at ``supersample``x then box-filters down to ``img_size`` for AA."""
    import numpy as np
    import trimesh
    from PIL import Image

    pyrender, renderer = _pyrender_renderer(img_size, supersample)
    V, F = _load_normalized_mesh(obj_path)          # raises -> caller scores benign

    os.makedirs(out_dir, exist_ok=True)
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(base_color, base_color, base_color, 1.0),
        metallicFactor=0.0, roughnessFactor=0.9)
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    mesh = pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False)  # flat-shaded facets
    cam = pyrender.OrthographicCamera(xmag=ortho_scale / 2.0, ymag=ortho_scale / 2.0,
                                      znear=0.01, zfar=100.0)
    key = pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=key_intensity)
    fill = pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=fill_intensity)
    up, origin = (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)

    paths = []
    for i, eye in enumerate(_camera_eyes(n_views, elevation_deg, dist)):
        az = 360.0 * i / n_views                    # this view's azimuth (matches _camera_eyes)
        scene = pyrender.Scene(bg_color=(bg, bg, bg, 1.0),
                               ambient_light=(ambient, ambient, ambient))
        scene.add(mesh)
        scene.add(cam, pose=_look_at(eye, origin, up))
        # Camera-relative studio rig: high raking key 35deg off-axis, soft low fill
        # 50deg the other way -- both swing with the camera so shading is view-consistent.
        scene.add(key, pose=_look_at(_eye_at(az + 35.0, 55.0, dist), origin, up))
        scene.add(fill, pose=_look_at(_eye_at(az - 50.0, 12.0, dist), origin, up))
        color, _ = renderer.render(scene)           # HxWx3 uint8 (supersampled)
        img = Image.fromarray(color[:, :, :3], "RGB")
        if supersample != 1:
            img = img.resize((img_size, img_size), Image.LANCZOS)
        out = os.path.join(out_dir, f"view{i}.png")
        img.save(out)
        paths.append(out)
    return paths


def render_views(obj_path, out_dir, n_views=8, img_size=128,
                 elevation_deg=25.0, dist=3.0, ortho_scale=2.4):
    """Render ``n_views`` shaded PNGs of ``obj_path`` around the azimuth.

    Writes ``view0.png`` .. ``view{n_views-1}.png`` into ``out_dir`` (created if
    needed) and returns their paths in order. Dispatches to the backend selected
    by ``RENDER_BACKEND`` (``pyrender`` default, or ``bpy``); both import
    their heavy module lazily. Raises ``RuntimeError`` for a mesh-less /
    unusable file so the caller can treat it as a degenerate (benign) input.
    """
    if RENDER_BACKEND == "pyrender":
        return _render_views_pyrender(obj_path, out_dir, n_views, img_size,
                                      elevation_deg, dist, ortho_scale)
    return _render_views_bpy(obj_path, out_dir, n_views, img_size,
                             elevation_deg, dist, ortho_scale)
