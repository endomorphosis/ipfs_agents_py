import {createOrbitDB, Identities, OrbitDBAccessController} from '@orbitdb/core'
import {createHelia} from 'helia'
import {EventEmitter} from "events";
import {createLibp2p} from 'libp2p'
import {identify} from '@libp2p/identify'
import {gossipsub} from '@chainsafe/libp2p-gossipsub'
import {bitswap} from '@helia/block-brokers'
import {tcp} from '@libp2p/tcp'
import {mdns} from '@libp2p/mdns'
import process from 'node:process'
import {LevelBlockstore} from 'blockstore-level'
import { LevelDatastore } from "datastore-level";
import { createRequire } from "module";
import { WebSocketServer } from 'ws'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { bootstrap } from '@libp2p/bootstrap'
import { floodsub } from '@libp2p/floodsub'
import { mplex } from '@libp2p/mplex'
import { kadDHT, removePublicAddressesMapper } from '@libp2p/kad-dht'
import { peerIdFromString } from '@libp2p/peer-id'
import { pubsubPeerDiscovery } from '@libp2p/pubsub-peer-discovery'
import { WebSocket } from 'ws';
import { webSockets } from '@libp2p/websockets';
import { webRTC } from '@libp2p/webrtc';
import { circuitRelayTransport } from '@libp2p/circuit-relay-v2'
import { all } from '@libp2p/websockets/filters'
import { ping } from '@libp2p/ping'
import { multiaddr } from 'multiaddr'

const require = createRequire(import.meta.url);
let bootstrappers = [
    '/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ',
    '/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN',
    '/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb',
    '/dnsaddr/bootstrap.libp2p.io/p2p/QmZa1sAxajnQjVM8WjWXoMbmPd7NsWhfKsPkErzpm9wGkp',
    '/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa',
    '/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt'
]
const ipfsLibp2pOptions = {
    addresses: {
        listen: ['/ip4/0.0.0.0/tcp/0']
    },
    transports: [
        tcp(),
        webSockets({
            // filter: all
        }),
        // webRTC(),
        circuitRelayTransport({
            discoverRelays: 1
        })
    ],
    streamMuxers: [
        yamux(),
        mplex()
    ],
    connectionEncryption: [
        noise()
    ],
    peerDiscovery: [
        mdns({
            interval: 20e3
        }),
        pubsubPeerDiscovery({
            interval: 1000
        }),
        bootstrap({
            list: bootstrappers
        })
    ],
    services: {
        lanDHT: kadDHT({
            protocol: '/ipfs/lan/kad/1.0.0',
            peerInfoMapper: removePublicAddressesMapper,
            clientMode: false
        }),
        pubsub:
            gossipsub({
                allowPublishToZeroPeers: true
            }),
        identify: identify(),
        ping: ping({
            protocolPrefix: 'ipfs', // default
        }),
    },
    connectionManager: {

    }
}
if (bootstrappers.length > 0) {
    ipfsLibp2pOptions.peerDiscovery.push(bootstrap({
        list: bootstrappers
    }))
}

EventEmitter.defaultMaxListeners = 64;

let ipfs
let orbitdb
let db

async function run(options) {
    process.env.LIBP2P_FORCE_PNET = "1"
    const argv = require('minimist')(process.argv.slice(2))
    let ipAddress
    let dbAddress
    let index
    let chunkSize
    let swarmName
    let port
    if (!argv.ipAddress && !Object.keys(options).includes('ipAddress')) {
        ipAddress = "127.0.0.1"
    } else if (Object.keys(options).includes('ipAddress')){
        ipAddress = options.ipAddress
    }
    else if (argv.ipAddress) {
        ipAddress = argv.ipAddress
    }
    else{
        ipAddress = "127.0.0.1"
    }
    
    if (!argv.swarmName && !options.swarmName) {
        console.error('Please provide a swarm Name');
        // process.exit(1);
        swarmName = "caselaw"
    }
    else if (Object.keys(options).includes('swarmName')) {
        swarmName = options.swarmName
    }
    else if (argv.swarmName) {
        swarmName = argv.swarmName
    }
    else{
        swarmName = "caselaw"
    }
    
    if (!argv.port && !Object.keys(options).includes('port')) {
        console.error('Please provide a port number');
        // process.exit(1);
        port = 50001
    }else if (Object.keys(options).includes('port')) {
        port = options.port
    }else if (argv.port) {
        port = argv.port
    }else{
        port = 50001
    }


    if (!argv.chunkSize && !Object.keys(options).includes('chunkSize')) {
        console.error('Please provide a chunk size');
        // process.exit(1);
        chunkSize = 8
    }else if (Object.keys(options).includes('chunkSize')) {
        chunkSize = options.chunkSize
    }else if (argv.chunkSize) {
        chunkSize = argv.chunkSize
    }
    else{
        chunkSize = 8
    }

    if (!argv.index && !Object.keys(options).includes('index')) {
        console.error('Please provide an index');
        // process.exit(1);
        index = 1
    }
    else if (Object.keys(options).includes('index')) {
        index = options.index
    }
    else if (argv.index) {
        index = argv.index
    }
    else{
        index = 1
    }

    process.on('SIGTERM', handleTerminationSignal);
    process.on('SIGINT', handleTerminationSignal);
    console.info('Script is running. Press CTRL+C to terminate.');
    const id =  index
    const libp2p = await createLibp2p({  addresses: {
        //listen: [`/ip4/${ipAddress}/tcp/0`]
        listen: ['/ip4/0.0.0.0/tcp/0']
        }, ...ipfsLibp2pOptions})
    const blockstore = new LevelBlockstore(`./ipfs/`+id+`/blocks`)
    const datastore = new LevelDatastore(`./ipfs/`+id+`/datastore`);
    ipfs = await createHelia({blockstore: blockstore, libp2p: libp2p, datastore: datastore, blockBrokers: [bitswap()]})
    const identities = await Identities({ ipfs, path: `./orbitdb/`+id+`/identities` })
    const identity = identities.createIdentity({ id })
    ipfs.libp2p.addEventListener("peer:connect", event => {
        console.log('connected', event.detail)
    })


    orbitdb = await createOrbitDB({ipfs: ipfs, identities, id: id, directory: `./orbitdb/`+id})

    db = await orbitdb.open(swarmName+"-"+index+"-of-"+chunkSize,
        {type: 'documents',
            AccessController: OrbitDBAccessController({ write: ["*"], sync: false}),
        })
    let oldHeads = await db.log.heads()
    console.debug(`${new Date().toISOString()} initial heads ${JSON.stringify(Array.from(oldHeads, h => h.payload))}`)
    await new Promise(r => setTimeout(r, 5000));
    await db.close()
    console.debug(`${new Date().toISOString()} opening db for sync`)
    db = await orbitdb.open(swarmName+"-"+index+"-of-"+chunkSize,
        {type: 'documents',
            AccessController: OrbitDBAccessController({ write: ["*"]}),
        })
    db.events.on('join', async (peerId, heads) => {
        for await (let entry of heads) {
            console.info(`peer ${peerId} joined with head ${JSON.stringify(entry.payload)}`)
        }
        if (oldHeads) {
            for (let hash of Array.from(oldHeads, h => h.hash)) {
                let it = db.log.iterator({gt: hash})
                for await (let entry of it) {
                    console.debug(`new startup entry ${JSON.stringify(entry.payload)}`)
                    oldHeads = [entry]
                }
            }
        }
    })
    console.info(`${new Date().toISOString()}running with db address ${db.address}`)
    const wss = new WebSocketServer({ port: port });
    wss.on('connection', (ws) => {
        const ip = ws._socket.remoteAddress;
        if (ip === '127.0.0.1' || ip === '::ffff:127.0.0.1') {
            console.log('New WebSocket connection');
            ws.on('message', (message) => {
                message = JSON.parse(message.toString());
                console.log('Received message:', message);
                let method = Object.keys(message)[0];
                let data = message[method];
                // Handle WebSocket messages here
                switch (method) {
                    case 'ping':
                        // Handle ping logic
                        let ping_peers = ipfs.libp2p.peerStore.store.datastore.data;
                        console.log('Ping peers:', ping_peers);
                        let ping_peers_list = [];
                        let ping_peers_time = {}
                        for (let peer of ping_peers) {
                            setTimeout(() => {
                                console.log('Pinging peer:', peer[0]);
                            }, 1000);
                            let begin = Date.now();
                            console.log('Pinging peer:', peer[0]);
                            ping_peers_list.push(peer[0]);
                            let end = Date.now();
                            ping_peers_time[peer[0]] = end - begin;
                        }                        
                        // let ping_peers_time_string = JSON.stringify({'pong' : ping_peers_time});
                        // ws.send(ping_peers_time_string);
                        break;

                    case 'peers' :
                        // Handle peers logic
                        let peers_ls = ipfs.libp2p.peerStore.store.datastore.data;
                        console.log('Ping peers:', peers_ls);
                        let peers_list = [];
                        let peers_time = {}
                        for (let peer of peers_ls) {
                            let begin = Date.now();
                            console.log('Pinging peer:', peer[0]);
                            peers_list.push(peer[0]);
                            let end = Date.now();
                            peers_time[peer[0]] = end - begin;
                        }

                        console.log('Peers:', peers_list);
                        ws.send(JSON.stringify({ "peers": peers_list }));
                        break;

                    case 'pubsub':
                        // Handle pubsub logic
                        let topic = data.topic;
                        let message = data.message;
                        ipfs.pubsub.publish(topic, message);
                        break;

                    case 'insert':
                        // Handle insert logic
                        let insertKey = Object.keys(data)[0];
                        let insertValue = data[insertKey];
                        let insertDoc = {_id: insertKey, content: insertValue};
                        console.log('Inserting data: ', insertKey, insertValue);
                        validate(insertValue).then((result) => {
                            if (result) {
                                db.put(insertDoc).then((results) => {
                                    console.log('Data inserted:', data);
                                    insertDoc = { value: {_id: insertKey, content: insertValue}, key: insertKey,  hash: results};
                                    ws.send(JSON.stringify({"insert": insertDoc}));
                                }).catch((error) => {
                                    console.error(JSON.stringify({'error': {'Error inserting data' : {'error' : error}}}));
                                    ws.send(JSON.stringify({'error': {'Error inserting data' : {'error' : error}}}));
                                });
                            }
                            else{
                                console.error(JSON.stringify({ 'error' : { 'Insert Data validation failed' : { "value" : insertValue }}}));
                                ws.send(JSON.stringify({ 'error' : { 'Insert Data validation failed' : { "value" : insertValue }}}));
                            }
                        });
                        break;
                    case 'update':
                        // Handle update logic
                        let update = data;
                        let updateKey = Object.keys(update)[0];
                        let updateValue = update[updateKey];
                        let updateDoc = { value: {"_id": updateKey, "content": updateValue}, "key": updateKey}; 
                        let docToUpdate = db.get(updateKey).then((doc) => {
                            validate(doc).then((result) => {
                                if (result == true && doc.key == updateKey && doc.value.content != updateDoc.value.content) {
                                    updateDoc = { "_id" : updateKey, "content": updateValue };
                                    db.put(updateDoc).then((results) => {
                                        updateDoc = { value: {"_id": updateKey, "content": updateValue}, "key": updateKey, "hash": results}; 
                                        console.log('Data updated:', updateDoc);
                                        ws.send(JSON.stringify({"update": updateDoc}));
                                    }).catch((error) => {
                                        console.error('Error updating data:', error);
                                        ws.send(json.stringify({'error' : { 'error updating data' : { 'doc': updateDoc, 'error': error}}}));
                                    });
                                }else if (result == true && doc.key == updateKey && doc.value.content == updateDoc.value.content){
                                    console.log('Data already up to date:', updateDoc);
                                    ws.send(JSON.stringify({'error' : { 'Data already up to date' : { 'doc': updateDoc , 'key': updateKey}}}));
                                }
                                else{
                                    console.error('Data validation failed:', doc, result);
                                    ws.send(JSON.stringify({'error' : {'Data validation failed' : {'doc' : updateDoc, 'result': result}}}));
                                }
                            }).catch((error) => {
                                console.error('Error updating data:', error);
                                ws.send(JSON.stringify({'error' : {'Error updating document' : {'error': error}}}));                      })
                        }).catch((error) => {
                            console.error('Error updating document:', error);
                            ws.send(JSON.stringify({'error' : {'Error updating document' : {'error': error}}}));
                        });
                        break;
                    case 'select':
                        // Handle select logic
                        let selectID = data;
                        let docToSelect = db.get(selectID).then((doc) => {
                            console.log('Selected document:', doc);
                            ws.send(JSON.stringify({'select': doc}));
                        }).catch((error) => {
                            console.error(JSON.stringify({'error':{'Error selecting document': {'selectID' : selectID ,  'error': error}}}));
                            ws.send(JSON.stringify({'error':{'Error selecting document': {'selectID' : selectID ,  'error': error}}}));
                        })
                        break;
                    case 'select_all':
                        // Handle select all logic
                        let select_all_docs = db.all().then((docs) => {
                            console.log('Selected all documents:', docs);
                            ws.send(JSON.stringify({'select_all': docs}));
                        }).catch((error) => {
                            console.error(json.stringify({ 'error' : { 'Error selecting all documents' : { 'error' : error }}}));
                            ws.send(json.stringify({ 'error' : { 'Error selecting all documents' : { 'error' : error }}}));
                        });
                        break;
                    case 'delete':
                        // Handle delete by ID logic
                        let deleteId = data;
                        
                        let docToDelete = db.get(deleteId).then((doc) => {
                            if (doc != undefined) {
                                db.del(deleteId).then((deletedDoc) => {
                                    console.log(JSON.stringify({'delete': doc}));
                                    ws.send(JSON.stringify({'delete': doc}));
                                }).catch((error) => {
                                    console.error(JSON.stringify({ 'error' : { 'Error deleting document': { 'error' : error, "doc": doc}}}));
                                    ws.send(JSON.stringify({ 'error' : { 'Error deleting document': { 'error' : error, "doc": doc}}}));
                                });
                            }
                            else{
                                console.error(JSON.stringify({'error': { 'Document not found' : deleteId }}));
                                ws.send(JSON.stringify({'error': { 'Document not found' : deleteId }}));
                            }
                        }).catch((error) => {
                            console.error(JSON.stringify({ 'error' : { 'Error deleting document' : { 'error' : error, "deleteId" : deleteId}}}))
                            ws.send(JSON.stringify({ 'error' : { 'Error deleting document' : { 'error' : error, "deleteId" : deleteId}}}));
                        });
                        break;
                    default:
                        console.log('Unknown message:', message);
                        break;
                }
            });

            // ipfs.libp2p.pubsub.on('message', (msg) => {
            //     console.log('Received pubsub message:', msg);
            //     ws.send(JSON.stringify(msg));
            // });

        } else {
            console.log('Unauthorized connection attempt:', ip);
            ws.send('Unauthorized connection');
            ws.close();
        }
    });
    // try {
    //     let ingest_port = port - 20000
    //     const wss2 = new WebSocketServer({ port: ingest_port })

    //     console.info(`${new Date().toISOString()} getting updates ...`)
    //     db.events.on('update', async (entry) => {
    //         console.debug(`new head entry op ${entry.payload.op} with value ${JSON.stringify(entry.payload.value)}`)
    //         if (oldHeads) {
    //             for (let hash of Array.from(oldHeads, h => h.hash)) {
    //                 let it = db.log.iterator({gt: hash, lte: entry.hash})
    //                 for await (let entry of it) {
    //                     console.debug(`new updated entry ${JSON.stringify(entry.payload)}`)
    //                     oldHeads = [entry]
    //                     wss2.send(JSON.stringify({ "ingest" : entry.payload }))
    //                 }
    //             }
    //         } else {
    //             let it = db.log.iterator({lte: entry.hash})
    //             for await (let entry of it) {
    //                 console.debug(`new updated entry ${JSON.stringify(entry.payload)}`)
    //                 oldHeads = [entry]
    //                 wss2.send(JSON.stringify({ "ingest" : entry.payload }))
    //             }
    //         }
    //     })
    //     console.info(`${new Date().toISOString()} searching result: `)
    //     let result = await db.query(data => {
    //         return data.content === "content 5000"
    //     })
    //     console.info(`${new Date().toISOString()} result: `, JSON.stringify(result))
    // }
    // catch (error) {
    //     console.log('Error connecting to ingest server:', error);
    // }
    // finally {
    //     console.log('Connected to ingest server');
    // }
}

async function validate(data) {
    return true
}

async function handleTerminationSignal() {
    console.info('received termination signal, cleaning up and exiting...');
    await db.close()
    await orbitdb.stop()
    await ipfs.stop()
    process.exit();
}

async function test() {
    let ipAddress = "127.0.0.1"
    let orbitdbAddress = undefined
    let index = 1
    let chunkSize = 8  
    let swarmName = "caselaw"
    let port = 50001

    let test = {
        ipAddress: ipAddress,
        orbitdbAddress: orbitdbAddress,
        index: index,
        chunkSize: chunkSize,
        swarmName: swarmName,
        port: port
    }
    return await run(test)
}

await run({})