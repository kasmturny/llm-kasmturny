
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_with_color(color: str, *text, sep=" ", end="\n"):
    '''打印带颜色的字符串'''
    print(color, *text, Color.END, sep=sep, end=end)


def print_red(*text, sep=" ", end="\n"):
    '''打印红色字符'''
    print_with_color(Color.RED, *text, sep=sep, end=end)


def print_green(*text, sep=" ", end="\n"):
    '''打印绿色字符'''
    print_with_color(Color.GREEN, *text, sep=sep, end=end)


def print_yellow(*text, sep=" ", end="\n"):
    '''打印黄色字符'''
    print_with_color(Color.YELLOW, *text, sep=sep, end=end)


def print_blue(*text, sep=" ", end="\n"):
    '''打印蓝色字符'''
    print_with_color(Color.BLUE, *text, sep=sep, end=end)