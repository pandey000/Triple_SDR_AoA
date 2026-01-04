import logging
from colorama import init, Fore, Style

def setup_logging(log_dir='logs', console_level='INFO', file_level='DEBUG'):
    init() # Colorama for Windows/Linux compatibility
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level))
    
    # Custom Formatter for colored terminal output
    class ColoredFormatter(logging.Formatter):
        COLORS = {'DEBUG': Fore.CYAN, 'INFO': Fore.GREEN, 
                  'WARNING': Fore.YELLOW, 'ERROR': Fore.RED}
        def format(self, record):
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            return super().format(record)

    console.setFormatter(ColoredFormatter('%(levelname)s: %(message)s'))
    logger.addHandler(console)
    return logger