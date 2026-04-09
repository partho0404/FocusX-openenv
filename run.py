from task import get_tasks

def main():
    tasks = get_tasks()

    print(f"[OPENENV] Loaded {len(tasks)} tasks", flush=True)

    for t in tasks:
        print(f"[TASK REGISTERED] {t['name']}", flush=True)

if __name__ == "__main__":
    main()
