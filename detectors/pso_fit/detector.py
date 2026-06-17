import subprocess
import os

from ..base import Detector

REPO_URL = os.getenv('PSO_REPO', 'https://github.com/carl-vbn/swarm-optimization-cpp')
REPO_NAME = REPO_URL.split('/')[-1].replace('.git', '')

def _install():
    print(f"Cloning PSO repository from {REPO_URL}...")
    subprocess.run(['git', 'clone', REPO_URL, REPO_NAME], check=True)

    print("Building solver...")
    subprocess.run(['bash', '-c', f'cd {REPO_NAME} && make -j$(nproc)'], check=True)

    print("PSO solver built successfully.")

def _eval(obj_path):
    output = subprocess.check_output(
        [f'out/solver', '--protrusion-mesh', os.path.abspath(obj_path), '--confidence', '--threads', '8', '--iterations', '200', '--particles', '1500'],
        text=True,
        cwd=os.path.join(os.getcwd(), REPO_NAME)
    )
    return float(output.strip())

class ParticleSwarm(Detector):
    def __init__(self):
        super().__init__()

        if not os.path.exists(REPO_NAME):
            try:
                _install()
            except subprocess.CalledProcessError as e:
                print(f"Error during PSO installation: {e}")
                raise RuntimeError("Failed to install PSO solver.")


    def eval(self, obj_path) -> dict:
        try:
            score = _eval(obj_path)
            return {"score": score}
        except subprocess.CalledProcessError as e:
            print(f"Error during PSO evaluation: {e}")
            raise RuntimeError("Failed to evaluate PSO solver.")
