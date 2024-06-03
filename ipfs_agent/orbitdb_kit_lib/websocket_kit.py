import websocket
import json
from urllib.parse import urlencode

class WebSocketClient:
	def __init__(self, master_url, meta):
		if meta is None:
			meta = {}
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
		self.ws = websocket.WebSocketApp(
			master_url,
			on_open=self.on_open,
			on_message=self.on_message,
			on_error=self.on_error,
			on_close=self.on_close,
		).run_forever(
			ping_interval=5,
			ping_timeout=2
		)
		self.state = {
	        'status': 'disconnected'
        }

	print('connecting to master')

	def connect(self, master_url, meta):

		self.ws = websocket.WebSocketApp(
			master_url,
			on_open=self.on_open,
			on_message=self.on_message,
			on_error=self.on_error,
			on_close=self.on_close,
		)
	
	def subscribe(self):
		self.ws.run_forever(
			ping_interval=5,
			ping_timeout=2
		)

	def set_handlers(self, **handlers):
		self.handlers = handlers
		
	def send(self, message):
		self.ws.send(json.dumps(message))
		
	def on_open(self, ws):
		print('connection accepted')
		self.send({
			'event': 'init',
			'status': self.state['status']
		})

	def on_close(self, ws, code, message):
		print('lost connection to master')
		self.state['status'] = 'disconnected'
		
	def on_error(self, ws, error):
		if isinstance(error, KeyboardInterrupt):
			print('socket closed and exiting gracefully')
			exit(0)

	def on_message(self, ws, message):
		try:
			payload = json.loads(message)
			command = payload['command']
			del payload['command']
			self.handlers[command](**payload)

		except Exception as e:
			print('error while handling message from master:')
			print(e)
			pass


if __name__ == '__main__':
	client = WebSocketClient('ws://127.0.0.1:50001', {})
	# subscribe = client.subscribe()