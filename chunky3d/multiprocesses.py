from multiprocessing.managers import BaseManager
from multiprocessing import Process, JoinableQueue
from time import sleep
import os
import signal
import dill


class ProcessPool:
    """
    Class which enables multiprocess calls to custom functions
    """

    class Shared:
        """
        Object shared between processes. Sync'd by the BaseManager
        """

        def __init__(self):
            self.clear()

        def get(self):
            return self.data

        def add(self, val):
            self.data.append(val)

        def clear(self):
            self.data = []

    def __init__(self, processes_count, *args, **kwargs):
        self.sleep_length = 2
        self.processes_count = processes_count
        self.queue_jobs = JoinableQueue()
        self.processes = []

        BaseManager.register('Shared', self.Shared)
        self.manager = BaseManager()
        self.manager.start()
        self.shared = self.manager.Shared()

        for i in range(self.processes_count):
            p = Process(target=self.make_pool_call)
            p.id = i
            p.start()
            self.processes.append(p)

    def make_pool_call(self):
        while True:
            item_pickled = self.queue_jobs.get()

            if item_pickled is None:
                self.queue_jobs.task_done()
                break

            item = dill.loads(item_pickled)
            call = item.get('call')
            args = item.get('args')
            kwargs = item.get('kwargs')

            try:
                result = call(*args, **kwargs)
                self.shared.add(result)

            except Exception as e:
                import traceback
                traceback.print_exc()
                os.kill(os.getpid(), signal.SIGUSR1)

            self.queue_jobs.task_done()

    def add_job(self, job):
        """
        :param: job: has to be a dilled dict:
                     {
                         'call': function_to_be_called_by_process,
                         'args': [],
                         'kwargs': {},
                     }
        """
        self.queue_jobs.put(job)

    def finish_pool_queue(self):
        while self.queue_jobs.qsize() > 0:
            sleep(self.sleep_length)
        for i in range(self.processes_count):
            self.queue_jobs.put(None)
        self.queue_jobs.join()
        self.queue_jobs.close()
        for p in self.processes:
            p.join()
        del self.processes[:]

    def get_pool_results(self):
        return self.shared.get()

    def clear_pool_results(self):
        self.shared.clear()
