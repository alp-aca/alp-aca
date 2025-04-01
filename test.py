import time

def main():
    t = time.time()
    from alpaca.citations import citations
    tf = time.time()
    print(tf-t)
    
if __name__ == '__main__':
    print(time.time(), __file__, __name__)
    main()
