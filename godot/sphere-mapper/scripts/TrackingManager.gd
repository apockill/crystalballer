extends Node

@export var gakken: Node3D
@export var camera: Node3D

var websocket = WebSocketPeer.new()
var face_url = "ws://localhost:6942/faces"

class FaceLocationPacket:
	var location: Vector3
	
	func _init(face_data: Dictionary):
		location = Vector3(
			face_data["location"][0],
			face_data["location"][1],
			face_data["location"][2]
		)

class FaceUpdatePacket:
	var face_locations: Array[FaceLocationPacket]
	
	func _init(json_data: Dictionary):
		face_locations = []
		for face_data in json_data["face_locations"]:
			face_locations.append(FaceLocationPacket.new(face_data))

func _ready():
	websocket.connect_to_url(face_url)

func _process(_delta):
	var faces = _poll_for_faces()
	
	# If no new faces came in, there's nothing to do
	if faces == null:
		return

	print("Got faces %s" % [faces.face_locations])

func _poll_for_faces() -> FaceUpdatePacket:
	# Poll the websocket and return a face packet if any were found
	websocket.poll()
	var state = websocket.get_ready_state()
	
	if state == WebSocketPeer.STATE_OPEN:
		while websocket.get_available_packet_count():
			var raw_packet_data = websocket.get_packet().get_string_from_utf8()
			var packet_data = parse_json(raw_packet_data)
			return FaceUpdatePacket.new(packet_data)

	elif state == WebSocketPeer.STATE_CONNECTING:
		print("Websocket is connecting!")		
	elif state == WebSocketPeer.STATE_CLOSING:
		# Keep polling to achieve proper close.
		print("Websocket is closing!")
	elif state == WebSocketPeer.STATE_CLOSED:
		var code = websocket.get_close_code()
		var reason = websocket.get_close_reason()
		print("WebSocket closed with code: %d, reason %s. Clean: %s" % [code, reason, code != -1])
		websocket = WebSocketPeer.new()
		websocket.connect_to_url(face_url)

	return null

func parse_json(data) -> Dictionary:
	var face_data = JSON.new()
	face_data.parse(data)
	return face_data.data


