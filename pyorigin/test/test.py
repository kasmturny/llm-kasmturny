import time


class ThreadPool:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.tasks = []
        self.workers = []

    def add_task(self, task):
        self.tasks.append(task)

    def run(self):
        while self.tasks or self.workers:
            # 添加新的worker
            while len(self.workers) < self.max_workers and self.tasks:
                task = self.tasks.pop(0)
                worker = Worker(task)
                self.workers.append(worker)

            # 执行worker任务
            for worker in self.workers:
                worker.run()

            # 移除
            self.workers = [worker for worker in self.workers if not worker.is_finished()]
            time.sleep(0.1)
# 使用示例
if __name__ == '__main__':
    pool = ThreadPool(max_workers=3)
    for i in range(10):
        pool.add_task(lambda: print(f'Task {i} is running'))
    pool.run()
    print('All tasks are finished')


