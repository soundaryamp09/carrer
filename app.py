from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

app = Flask(__name__)
CORS(app)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key="pcsk_4VmWuq_Djoy44geTPCJU2bxN1wg5PnDB6CjBr7V2iDTB5KTcVj6qfMG8FkGVsZDmu3BHAa")
index = pc.Index("career-recommendation")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    text = f"{data.get('education','')}. {data.get('skills','')}. {data.get('interests','')}. {data.get('goal','')}"
    vec = model.encode(text).tolist()
    resp = index.query(vector=vec, top_k=1, include_metadata=True)
    if not resp['matches']:
        return jsonify({"recommendation": "No match found"}), 404
    rec = resp['matches'][0]['metadata'].get("career", "Unknown Career")
    return jsonify({"recommendation": rec})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
