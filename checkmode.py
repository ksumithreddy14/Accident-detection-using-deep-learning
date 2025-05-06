import tensorflow as tf

# Path to the frozen model
model_path = "model/frozen_inference_graph.pb"

# Load the model
with tf.io.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Look for key model components
feature_extractors = set()
for node in graph_def.node:
    if "feature_extractor" in node.name:
        feature_extractors.add(node.name)

# Print results
if feature_extractors:
    print("Detected Feature Extractors in the Model:")
    for extractor in feature_extractors:
        print("-", extractor)
else:
    print("Feature extractor not found. The model might be an SSD, Faster R-CNN, or a YOLO-based architecture.")
