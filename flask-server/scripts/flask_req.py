import requests
import json


payload = {
	"content": "yacht charter yacht rentals boat charters yachtico com iframe src https www googletagmanager com ns html idgtm fbm height width style display none visibility hidden iframe any questions call deutsch english yacht charters rentals caribbean top yacht charters caribbean antigua and barbuda bahamas yacht charter us virgin island charters cuba yacht charter belize yacht charter british virgin islands bvi barbados yacht charters saint vincent and the grenadines mediterranean top yacht charters mediterranean croatia yacht charter greece yacht charter italy yacht charter france charters asia africa north america asia africa north america malaysia yacht charter seychelles yacht charters cape verde charters brazil yacht charter mexico yacht charters australien and oceanien australia oceania french polynesia new caledonia charters new zealand yacht charter yacht charters rentals worldwide yacht vacations bareboat charter luxury crewed charters powerboat or motoryacht cruising houseboating sailboat charters catamaran charters sailing destinations itineraries charter guide world s yacht charter yacht rentals since more than charter yachts worldwide to book online location check in check out what are you looking for all yachts sailing boat motorboat catamaran houseboat all yachts all yachts sailing boat motorboat catamaran houseboat professionally maintained ",
	"metaDescription": "yacht charter yacht rentals ready to book a yacht charter vacation online charter yachts worldwide bareboat charter crewed yacht charters sailboats sailing and motoryachts catamarans houseboats",
	"metaKeywords": "crewed yacht charter skippered yacht charters bareboat sailing yacht rentals rent a yacht boat charter sailing charter sailboat motoryacht catamaran rental chartering hire"
}

r = requests.post('http://localhost:8600/getPrediction', json=payload)

pred = json.loads(r.content.decode('utf-8'))

print(pred)