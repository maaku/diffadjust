#!/usr/bin/env python3

import codecs
from bitcoin.rpc import RawProxy

if __name__ == '__main__':
    conf = input('Please enter path to the configuration file: ')

    proxy = RawProxy(btc_conf_file=conf)
    numblocks = proxy.getblockcount() + 1

    fp = codecs.open('out.csv', 'w', 'utf-8')

    for idx in range(numblocks):
        blockinfo = proxy.getblock(proxy.getblockhash(idx))
        fp.write(','.join(map(str, [
            blockinfo['height'],
            blockinfo['time'],
            blockinfo['difficulty'],
        ]))+'\n')

    fp.close()

# End of File
