import os

def get_available_cpus():
    # Check for SLURM environment variables (for HPC servers using Slurm)
    if "SLURM_CPUS_PER_TASK" in os.environ and "SLURM_NTASKS" in os.environ:
        cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
        num_tasks = int(os.environ["SLURM_NTASKS"])
        return cpus_per_task * num_tasks
    
    # Fallback to detecting all available CPUs (for local PCs or when not using a scheduler)
    return os.cpu_count()