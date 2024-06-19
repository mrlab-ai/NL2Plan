import os, datetime, json, logging

from .paths import results_dir

class LoggerClass:
    def __init__(self):
        self.started = False

    def start(self, domain, **kwargs):
        self.started = True

        self.domain = domain
        self.directory = os.path.join(results_dir, self.domain, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.name = os.path.join(self.domain, os.path.basename(self.directory))
        self.info_file = os.path.join(self.directory, "info.log")

        # Create directory and info file
        os.makedirs(self.directory)
        with open(self.info_file, "w") as file:
            file.write(f"Domain: {domain}\n")
            file.write(f"Time: {datetime.datetime.now()}\n")
            for k, v in kwargs.items():
                file.write(f"{k}: {v}\n")

        # Create logger
        self.logger = logging.getLogger("LLM Planner")
        self.logger.setLevel(logging.DEBUG) # Set to lowest level
        self.formatter = logging.Formatter('%(message)s') # Message only

    def section(self, name: str):
        def inner_decorator(func):
            def wrapper(*args, **kwargs):
                # Check if logger is started
                if not self.started:
                    print("Warning: Logger not set up, call logger.start() to init.")
                    return func(*args, **kwargs)
                
                # Set up section handlers
                self.logger.handlers.clear() # Clear all handlers
                # File handler
                file_path = os.path.join(self.directory, f"{name.replace(' ', '_')}.log")
                file_handler = logging.FileHandler(file_path)
                file_handler.setLevel(logging.DEBUG) # Everything is logged in file
                file_handler.setFormatter(self.formatter)
                self.logger.addHandler(file_handler)
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO) # Only some is logged in console
                console_handler.setFormatter(self.formatter)
                self.logger.addHandler(console_handler)

                self.print(f"{'#'*10} {name} {'#'*10}")
                res = func(*args, **kwargs)
                self.print(f"{'#'*30}")
                return res
            return wrapper
        return inner_decorator

    def print(self, *messages: list[str], sep: str = '', subsection: bool = True):
        """Print message to log file and console. Should be used within a section."""
        message = sep.join([str(m) for m in messages])
        if subsection: message = f"{'-'*100}\n{message}\n{'-'*100}"
        if not self.started: print(message)
        else: self.logger.info(message)
    
    def log(self, *messages: list[str], sep: str = '', subsection: bool = True):
        """Log message to log file. Should be used within a section."""
        message = sep.join([str(m) for m in messages])
        if subsection: message = f"{'-'*100}\n{message}\n{'-'*100}"
        if not self.started: print(message)
        else: self.logger.debug(message)

    def add_to_info(self, **kwargs):
        """Add key-value pairs to info file. Underscores in keys are replaced with spaces."""
        with open(self.info_file, "a") as file:
            for k, v in kwargs.items():
                file.write(f"{k.replace('_',' ')}: {v}\n")

    def add_domain_desc(self, domain_desc: dict):
        """Add domain description to info file."""
        desc = f"\n{json.dumps(domain_desc, indent=2)}"
        self.add_to_info(Domain_description = desc)

Logger = LoggerClass() # Global logger