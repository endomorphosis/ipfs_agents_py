import orbitdb_kit_lib as orbitdb_kit
import ipfs_model_manager as ipfs_model_manager
import libp2p_kit as libp2p_kit
# import ipfs_accelerate as ipfs_accelerate
import config as config
import os
import asyncio

class ipfs_agent:
    def __init__(self, resources=None, meta=None):
        self.ls_models = {}
        self.tools = {}
        self.peers = {}
        self.libp2p_services = {}
        if meta is None:
            meta = {}
            meta['config'] = 'config/config.toml'
        else:
            if 'config' not in meta:
                meta['config'] = 'config/config.toml'
            else:
                meta['config'] = meta['config']
        if resources is None:
            resources = {}
        self.config = config.config({}, meta=meta)
        self.baseConfig = self.config.baseConfig
        for key, value in self.baseConfig.items():
            meta[key] = value
        self.model_manager = ipfs_model_manager.ipfs_model_manager(resources, meta)
        self.orbitdb_kit = orbitdb_kit.orbitdb_kit(resources, meta)
        self.libp2p_kit = libp2p_kit.libp2p_kit(resources, meta)
        # self.accelerate = ipfs_accelerate.ipfs_accelerate(resources, meta)
        pass

    async def __call__(self):
        config = self.config
        self.model_manager.ls_models()
        #models = self.model_manager.list_ipfs_models()
        # orbitdb = self.orbitdb_kit.stop_orbitdb()
        # orbitdb = self.orbitdb_kit.start_orbitdb()
        # asyncio.run(self.orbitdb_kit.connect_orbitdb())
        #print("connect")
        #print(connect)
        #results = asyncio.get_event_loop().run_until_complete(self.orbitdb_kit.connect_orbitdb())
        ws =  await self.orbitdb_kit.connect_orbitdb(self.callbackFunction())
        return ws
        insert_key = { "test": "test document" }
        insert = await self.orbitdb_kit.insert_request(insert_key)
        print("insert")
        print(insert)
        # update_key = { "test": "test document @ update" }
        # update = self.orbitdb_kit.update_orbitdb(update_key)
        # print("update")
        # print(update)
        # select_key = "test"
        # select = self.orbitdb_kit.select_orbitdb(select_key)
        # print("select")
        # print(select)
        # delete_key = "test"
        # delete = self.orbitdb_kit.delete_orbitdb(delete_key)
        # print("delete")
        # print(delete)
        # orbitdb = self.orbitdb_kit.stop_orbitdb()
        results = {
            "config": config.baseConfig,
            "orbitdb": self.orbitdb_kit.orbitdb
            # "connect": connect,
            # "insert": insert,
            # "update": update,
            # "select": select,
            # "delete": delete
        }
        return results
    
    async def run(self):
        return await self.__call__()
    
    async def callbackFunction(self, *args, **kwargs):

        print("callbackFunction")
        return True
    
if __name__ == '__main__':
    meta = {}
    resources = {}
    ipfs_agent = ipfs_agent(resources, meta)
    # results = asyncio.get_event_loop().run_until_complete(ipfs_agent.run())
    results = asyncio.run(ipfs_agent.run())
    # print(results)
    # print(ipfs_agent())