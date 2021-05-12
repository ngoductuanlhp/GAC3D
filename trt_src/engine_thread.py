import threading
import pycuda.driver as cuda
from engine import Engine

class EngineThread(threading.Thread):
    """EngineThread
    This implements the child thread which continues to read images
    from cam (input) and to do TRT engine inferencing.  The child
    thread stores the input image and detection results into global
    variables and uses a condition varaiable to inform main thread.
    In other words, the TrtThread acts as the producer while the
    main thread is the consumer.
    """
    def __init__(self, eventStart, eventEnd, args):
        threading.Thread.__init__(self)
        
        self.eventStart = eventStart
        self.eventEnd = eventEnd
        self.args = args
        
        self.eventStop = threading.Event()

        self.cuda_ctx = None  # to be created when run
        self.engine = None   # to be created when run

        self.img = None

        self.hm, self.features = None, None


    def run(self):
        """Run until 'running' flag is set to False by main thread.
        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """

        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.engine = Engine(self.args.load_model, self.args.dcn_lib)

        while not self.eventStop.is_set():
            self.eventStart.wait()
            self.eventStart.clear()
            if self.img is not None:
                self.cuda_ctx.push()
                self.hm, self.features = self.engine(self.img)
                self.cuda_ctx.pop()
                self.img = None
            self.eventEnd.set()

        del self.engine
        self.cuda_ctx.pop()
        del self.cuda_ctx

    def stop(self):
        self.eventStop.set()