# !pip install sentence-transformers
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_similarity_sbert(text1, text2):
    """
    This function calculates the cosine similarity between two texts using SBERT.
    """
    # Encode both texts
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

# Vercel handler
def handler(req, res):
    if req.method == 'POST':
        if not req.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = req.get_json()
        string1 = data.get('text1')
        string2 = data.get('text2')

        # Calculate similarity
        similarity = calculate_similarity_sbert(string1, string2)
        similarity = max(0, similarity)

        return jsonify({'similarity': round(similarity, 4)})
    else:
        return jsonify({"error": "Method not allowed"}), 405

# Entry point for Vercel
if __name__ == '__main__':
    app.run()

# Vercel needs to expose a callable for the serverless function
def vercel_function(req, res):
    return handler(req, res)
