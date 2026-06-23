import multiprocessing
import os


def grandchild_task(name):
    print(f"Grandchild {name} (PID: {os.getpid()}) is running.")


def child_task(name):
    print(f"Child {name} (PID: {os.getpid()}) is running.")

    # In a nested scenario, we create a new context to ensure
    # the grandchild also uses the forkserver.
    ctx = multiprocessing.get_context('forkserver')

    p = ctx.Process(target=grandchild_task, args=(f"GC-{name}",))
    p.start()
    p.join()


if __name__ == '__main__':
    # 1. Set the global start method to forkserver
    # This must be inside the 'if __name__ == "__main__":' block.
    multiprocessing.set_start_method('forkserver', force=True)

    print(f"Main Process (PID: {os.getpid()})")

    # 2. Spawn the first level child
    parent_process = multiprocessing.Process(target=child_task, args=("C1",))
    parent_process.start()
    parent_process.join()