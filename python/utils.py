from termcolor import colored

def progress_bar(current, total, bar_length=50, start_time=None, curr_time=None):
    fraction = current / total
    arrow = int(fraction * bar_length) * '—'
    padding = (bar_length - len(arrow)) * '—'
    if start_time and curr_time:
        ending = '\n' if current == total else '\r'
        print('Progress: ['+colored(f'{arrow}','green')+colored(f'{padding}','red')+f'] {round(fraction*100,1)}% Time elapsed: {round(curr_time-start_time,5)}s', end=ending)
    else:
        ending = '\n' if current == total else '\r'
        print('Progress: ['+colored(f'{arrow}','green')+colored(f'{padding}','red')+f'] {round(fraction*100,1)}%', end=ending)