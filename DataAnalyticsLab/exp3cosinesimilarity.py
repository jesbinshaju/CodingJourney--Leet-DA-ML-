import math
import string

# ----------------------------------------
# Input Documents
# ----------------------------------------
doc1 = input("Enter Document 1: ")
doc2 = input("Enter Document 2: ")


#If file 

# ----------------------------------------
# Read Documents from Files
# ----------------------------------------
with open("document1.txt", "r") as file:
    doc1 = file.read()

with open("document2.txt", "r") as file:
    doc2 = file.read()


# ----------------------------------------
# Preprocessing Function
# ----------------------------------------
def preprocess(text):
    text = text.lower()

    # Remove punctuation
    for ch in string.punctuation:
        text = text.replace(ch, "")

    words = text.split()
    return words

# ----------------------------------------
# Create Vocabulary
# ----------------------------------------
words1 = preprocess(doc1)
words2 = preprocess(doc2)

vocabulary = sorted(set(words1 + words2))

# ----------------------------------------
# Create Term Frequency Vector
# ----------------------------------------
def term_frequency(words, vocabulary):
    vector = []

    for word in vocabulary:
        vector.append(words.count(word))

    return vector

vector1 = term_frequency(words1, vocabulary)
vector2 = term_frequency(words2, vocabulary)

# ----------------------------------------
# Dot Product
# ----------------------------------------
dot_product = 0

for i in range(len(vocabulary)):
    dot_product += vector1[i] * vector2[i]

# ----------------------------------------
# Magnitudes
# ----------------------------------------
magnitude1 = 0
magnitude2 = 0

for value in vector1:
    magnitude1 += value ** 2

for value in vector2:
    magnitude2 += value ** 2

magnitude1 = math.sqrt(magnitude1)
magnitude2 = math.sqrt(magnitude2)

# ----------------------------------------
# Cosine Similarity
# ----------------------------------------
if magnitude1 == 0 or magnitude2 == 0:
    cosine_similarity = 0
else:
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

# ----------------------------------------
# Output
# ----------------------------------------
print("\nVocabulary:")
print(vocabulary)

print("\nVector 1:")
print(vector1)

print("\nVector 2:")
print(vector2)

print("\nDot Product:", dot_product)
print("Magnitude of Vector 1:", magnitude1)
print("Magnitude of Vector 2:", magnitude2)

print("\nCosine Similarity =", round(cosine_similarity, 4))

if cosine_similarity > 0.8:
    print("Documents are Highly Similar.")
elif cosine_similarity > 0.5:
    print("Documents are Moderately Similar.")
else:
    print("Documents are Less Similar.")
   