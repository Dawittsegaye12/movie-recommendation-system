import sys
#import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail)

    def error_message_detail(self, error_message, error_detail: sys):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno

        error_message_detail = f"The error occurred in {file_name} on line number {line_no}. The error is: {error_message}"
        return error_message_detail

    def __str__(self):
        return self.error_message
    
    print("hello world")