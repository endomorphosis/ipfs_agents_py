import os
import sys
import subprocess as process 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import websockets as ws 
import asyncio
from config import config
import json
from .websocket_kit import WebSocketClient as WebSocketClient
import datetime
import time

class orbitdb_kit():
    def __init__(self,  resources=None, meta=None):
        self.resources = resources
        self.meta = meta
        self.config = config
        self.connection = None
        self.orbitdb_args = {}
        self.peers = []
        self.ping = {}
        self.state = {
            "status": "disconnected"
        }
        self.hash_list = []
        self.key_list = []
        self.key_hash_dict = {}
        self.orbitdb = []
        self.ws = None
        self.url = None
        self.orbitdb_args['ipaddress'] = None
        self.orbitdb_args['orbitdbAddress'] = None
        self.orbitdb_args['index'] = None
        self.orbitdb_args['chunkSize'] = None
        self.orbitdb_args['swarmName'] = None
        self.orbitdb_args['port'] = None
        self.this_dir = os.path.dirname(os.path.realpath(__file__))
        if self.meta is None:
            self.meta = {}
            self.on_open = self.on_open
            self.on_message = self.on_message
            self.on_error = self.on_error
            self.on_close = self.on_close
        else:
            if 'on_open' not in meta:
                self.on_open = self.on_open
            else:
                self.on_open = meta['on_open']
            if 'on_message' not in meta:
                self.on_message = self.on_message
            else:
                self.on_message = meta['on_message']
            if 'on_error' not in meta:
                self.on_error = self.on_error
            else:
                self.on_error = meta['on_error']
            if 'on_close' not in meta:
                self.on_close = self.on_close
            else:
                self.on_close = meta['on_close']
            if "orbitdb" in meta:
                if 'ipAddress' in meta["orbitdb"] and meta["orbitdb"]["ipAddress"] is not None and meta["orbitdb"]["ipAddress"] != '':
                    self.orbitdb_args['ipaddress'] = meta["orbitdb"]["ipAddress"]
                if 'orbitdbAddress' in meta["orbitdb"] and meta["orbitdb"]["orbitdbAddress"] is not None and meta["orbitdb"]["orbitdbAddress"] != '':
                    self.orbitdb_args['orbitdbAddress'] = meta["orbitdb"]["orbitdbAddress"]
                if 'index' in meta["orbitdb"] and meta["orbitdb"]["index"] is not None and meta["orbitdb"]["index"] != '':
                    self.orbitdb_args['index'] = meta["orbitdb"]["index"]
                if 'chunkSize' in meta["orbitdb"] and meta["orbitdb"]["chunkSize"] is not None and meta["orbitdb"]["chunkSize"] != '':
                    self.orbitdb_args['chunkSize'] = meta["orbitdb"]["chunkSize"]
                if 'swarmName' in meta["orbitdb"] and meta["orbitdb"]["swarmName"] is not None and meta["orbitdb"]["swarmName"] != '':
                    self.orbitdb_args['swarmName'] = meta["orbitdb"]["swarmName"]
                if 'port' in meta["orbitdb"] and meta["orbitdb"]["port"] is not None and meta["orbitdb"]["port"] != '':
                    self.orbitdb_args['port'] = meta["orbitdb"]["port"]
            else:
                pass
            if 'ipaddress' in self.meta and self.meta['ipaddress'] is not None and self.meta['ipaddress'] != '':
                self.orbitdb_args['ipaddress'] = self.meta['ipaddress']           
            if 'orbitdbAddress' in self.meta and self.meta['orbitdbAddress'] is not None and self.meta['orbitdbAddress'] != '':
                self.orbitdb_args['orbitdbAddress'] = self.meta['orbitdbAddress']            
            if 'index' in self.meta and self.meta['index'] is not None and self.meta['index'] != '':
                self.orbitdb_args['index'] = self.meta['index']            
            if 'chunkSize' in self.meta and self.meta['chunkSize'] is not None and self.meta['chunkSize'] != '':
                self.orbitdb_args['chunkSize'] = self.meta['chunkSize']
            else:
                self.orbitdb_args['chunkSize'] = None
            if 'swarmName' in self.meta and self.meta['swarmName'] is not None and self.meta['swarmName'] != '':
                self.orbitdb_args['swarmName'] = self.meta['swarmName']
            if 'port' in self.meta and self.meta['port'] is not None and self.meta['port'] != '':
                self.orbitdb_args['port'] = self.meta['port']
        
        if self.orbitdb_args['ipaddress'] is None:
            self.orbitdb_args['ipaddress'] = '127.0.0.1'
        if self.orbitdb_args['orbitdbAddress'] is None:
            self.orbitdb_args['orbitdbAddress'] = None
        if self.orbitdb_args['index'] is None:
            self.orbitdb_args['index'] = 1
        if self.orbitdb_args['chunkSize'] is None:
            self.orbitdb_args['chunkSize'] = 8
        if self.orbitdb_args['swarmName'] is None:
            self.orbitdb_args['swarmName'] = "caselaw"
        if self.orbitdb_args['port'] is None:
            self.orbitdb_args['port'] = 50001
        pass

    def start_orbitdb(self , args = None):
        start_args = self.orbitdb_args
        if args is not None:
            for key, value in args.items():
                start_args[key] = value
        start_argstring = ''
        for key, value in start_args.items():
            start_argstring += ' --' + key + '=' + str(value) + ' '
        start_cmd = 'node ' + os.path.join(self.this_dir, 'orbitv3-slave-swarm.js') + ' ' + start_argstring  
        print(start_cmd)
        start_cmd = start_cmd.split(' ')
        start_orbitdb = process.Popen(start_cmd)
        # start_orbitdb = process.Popen(start_cmd, shell=True)
        # start_orbitdb = process.run(start_cmd, stdout=process.PIPE, stderr=process.PIPE)
        # pause for 5 seconds to allow orbitdb to start
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(5))
        asyncio.get_event_loop().run_until_complete(self.connect_orbitdb())
        return start_orbitdb
        pass

    async def connect_orbitdb(self, callback_fn = None):
        self.url = 'ws://' + self.orbitdb_args['ipaddress'] + ':' + str(self.orbitdb_args['port'])

        if (callback_fn is not None) and (callable(callback_fn)):
            self.ws = WebSocketClient(self.url, 
                {
                "on_open" : self.on_open(callback_fn),
                "on_message" : self.on_message,
                "on_error" : self.on_error,
                "on_close" : self.on_close
                }
            )
            return self.ws
        else:
            self.ws = WebSocketClient(self.url, 
                {
                "on_open" : self.on_open,
                "on_message" : self.on_message,
                "on_error" : self.on_error,
                "on_close" : self.on_close
                }
            )
            return self.ws
    
    def on_pong_message(self, ws, message):
        self.pong = message['pong']
        pass

    def on_ping_message(self, ws, recv):
        self.ping = recv['ping']
        pass

    def on_peers_message(self, ws, recv):
        self.peers = recv['peers']
        return self.peers
    
    def orbitdb_hash_list(self, ws, recv):
        hash_list = list(map(lambda x: x['hash'], self.orbitdb))
        return hash_list
    
    def orbitdb_key_list(self, ws, recv):
        key_list = list(map(lambda x: x['key'], self.orbitdb))
        return key_list
    
    def orbitdb_key_hash_dict(self, ws, recv):
        key_hash_dict = dict(map(lambda x: (x['key'], x['hash']), self.orbitdb))
        return key_hash_dict

    def on_insert_handler(self, ws, recv):
        hash_list = self.hash_list
        insert = recv['insert']
        hash = insert['hash']
        key = insert['key']
        if hash in hash_list:
            raise Exception("hash already exists", hash)
        if key in self.key_list:
            raise Exception("key already exists", key)

        if hash not in hash_list and key not in self.key_list:
            self.hash_list.append(hash)
            self.key_list.append(key)
            self.key_hash_dict[key] = hash
            self.orbitdb.append(insert)
            
        return insert
    
    def on_update_handler(self, ws, recv):
        self.hash_list = self.orbitdb_hash_list(ws, recv)
        self.key_list = self.orbitdb_key_list(ws, recv)
        self.key_hash_dict = self.orbitdb_key_hash_dict(ws, recv)
        update = recv['update']
        update_key = update['key']
        if update_key in self.key_list:
            rm_hash = self.key_hash_dict[update_key]
            rm_index = self.key_list.index(update_key)
            self.orbitdb.pop(rm_index)
            self.orbitdb.append(update)
            del(self.key_hash_dict[update_key])
            self.hash_list.pop(self.hash_list.index(rm_hash))
            self.key_list.pop(self.key_list.index(update_key))
            self.key_hash_dict[update_key] = update['hash']
            self.hash_list.append(update['hash'])
            self.key_list.append(update_key)
        else:
            raise Exception("key does not exist")
        
        return update
    
    def on_select_handler(self, ws, recv):
        select = recv['select']
        select_key = select['key']
        select_hash = select["hash"]
        hash_list = self.hash_list
        key_list = self.key_list
        if select_hash not in hash_list and select_key in key_list:
            self.hash_list.append(select_hash)
            index = key_list.index(select_key)
            self.orbitdb.pop(index)
            self.orbitdb.append(select)
            # remove old hash and append new one
        if select_key not in self.key_list and select_hash in self.hash_list:
            self.key_list.append(select_key)
            self.key_hash_dict[select_key] = select_hash
            self.orbitdb.append(select)
        if select_key not in self.key_list and select_hash not in self.hash_list:
            self.key_list.append(select_key)
            self.hash_list.append(select_hash)
            self.key_hash_dict[select_key] = select_hash
            self.orbitdb.append(select)
        if select_key in self.key_list and select_hash in self.hash_list:
            index = key_list.index(select_key)
            self.orbitdb.pop(index)
            self.orbitdb.append(select)
            hash_list.pop(hash_list.index(select_hash))
            key_list.pop(key_list.index(select_key))
            pass
        return select

    def on_delete_handler(self, ws, recv):
        delete_hash = recv['delete']['hash']
        delete_key = recv['delete']['key']
        if delete_hash in self.hash_list and delete_key in self.key_list:
            key_hash = self.key_hash_dict[delete_key]
            orbit_db_index = self.hash_list.index(key_hash)
            self.orbitdb.pop(orbit_db_index)
            self.key_hash_dict.__delitem__(delete_key)
            self.hash_list.pop(self.hash_list.index(delete_hash))
            self.key_list.pop(self.key_list.index(delete_key))
            return { 'delete' : {"hash": delete_hash, "key": key_hash}}
        else:
            raise Exception("hash does not exist")

    def on_select_all_handler(self, ws, recv):
        self.orbitdb = recv['select_all']
        self.hash_list = self.orbitdb_hash_list(ws, recv)
        self.key_list = self.orbitdb_key_list(ws, recv)
        self.key_hash_dict = self.orbitdb_key_hash_dict(ws, recv)
        return self.orbitdb

    def on_message(self, ws, message):
        # print(f"Received message: message = '{message}')")
        recv = json.loads(message)
        results = ""

        if "error" in recv:
            results = self.on_error(
                ws, recv['error']
            )

        if 'pong' in recv:
            results = self.on_pong_message(
                ws, recv
            )
            
        if 'ping' in recv:
            results = self.on_ping_message(
                ws, recv
            )

        if 'peers' in recv:
            results = self.on_peers_message(
                ws, recv
            )
        
        if 'insert' in recv:
            results = self.on_insert_handler(
                ws, recv
            )
        
        if 'select_all' in recv:
            results = self.on_select_all_handler(
                ws, recv
            )
        
        if 'update' in recv:
            results = self.on_update_handler(
                ws, recv
            )
        
        if 'delete' in recv:
            results = self.on_delete_handler(
                ws, recv
            )

        if 'select' in recv:
            results = self.on_select_handler(
                ws, recv
            )
        
        # print("results",  results)
        return results
    
    def on_error(self, ws, error):
        print(f"Error occurred: {error}")
        return error

    def on_close(self, ws, arg1, arg2):
        print("Connection closed")
        return ws
    
    def peers_ls_request(self, ws):
        ws.send(json.dumps({
            'peers': 'ls'
        }))
        return self.peers
    
    def select_all_request(self, ws):
        ws.send(json.dumps({
            'select_all':  "*"
        }))
        return True
    
    def insert_request(self, ws, data):
        data_keys = list(data.keys())
        if type(data) is not dict:
            raise Exception("data must be a dictionary")
        if len(data_keys) != 1:
            raise Exception("data must have exactly one key")
        ws.send(json.dumps({
            'insert': data
        }))
        return True
    
    def update_request(self, ws, data):
        data_keys = list(data.keys())
        if type(data) is not dict:
            raise Exception("data must be a dictionary")
        if len(data_keys) != 1:
            raise Exception("data must have exactly one key")
        ws.send(json.dumps({
            'update': data
        }))
        return True
    
    def delete_request(self, ws, data):
        if type(data) is not str:
            raise Exception("data value must be a string")
        ws.send(json.dumps({
            'delete': data
        }))
        return True

    def select_request(self, ws, data):
        if type(data) is not str:
            raise Exception("data value must be a string")
        ws.send(json.dumps({
            'select': data
        }))
        return True
    
    def on_open(self, ws, callback_fn = None):
        print('connection accepted')
        print("url", self.url)
        peers = self.peers_ls_request(ws)
        select_all = self.select_all_request(ws)
        # insert = self.insert_request(ws, {"test": "test document"})
        # update = self.update_request(ws, {"test": "update document"})
        # select = self.select_request(ws, "test")
        # delete = self.delete_request(ws, "test")
        if callback_fn is not None:
            results = callback_fn(ws)
        else:
            results = ws
        return results


    def stop_orbitdb(self, args = None):
        ps_orbitb_results = None
        stop_orbitdb_results = None

        start_args = self.orbitdb_args
        if args is not None:
            for key, value in args.items():
                start_args[key] = value
        start_argstring = ''
        ps_orbitdb = 'ps -ef | grep orbitdb | grep -v grep | awk \'{print $2}\' | grep port=' + str(start_args['port']) + " "
        stop_orbitdb = 'ps -ef | grep orbitdb | grep -v grep | awk \'{print $2}\' | grep port=' + str(start_args['port']) + ' | xargs kill -9'
        print(ps_orbitdb)
        print(stop_orbitdb)
        try:
            ps_orbitb_results = process.check_output(ps_orbitdb, shell=True)
        except Exception as e:
            print(e)
            ps_orbitb_results = e
            pass
        finally:
            if ps_orbitb_results is not None and ( type == bytes or type == str):
                try:
                    stop_orbitdb_results = process.check_output(stop_orbitdb, shell=True)
                except Exception as e:
                    print(e)
                    raise e
                finally:
                    pass

        results = {
            'ps_orbitdb': ps_orbitb_results,
            'stop_orbitdb': stop_orbitdb_results
        }
        return results

    def get_resources(self):
        return self.resources
    
    def main(self):
        # print("websocket client main()")
        # print("main loop ()")
        # while True:
        #     time.sleep(5)
        #     pass
        return True

    async def test(self):
        print("websocket client test()")
        await orbitdb_kit.connect_orbitdb()
    
if __name__ == '__main__':
    resources = {}
    meta = {}
    orbitdb_kit = orbitdb_kit(resources, meta)
    results = asyncio.run(orbitdb_kit.test())
    print("done")

