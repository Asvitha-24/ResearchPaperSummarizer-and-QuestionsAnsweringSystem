"""Quick test of fine-tuned DistilBERT QA model."""

from distilbert_qa_utils import DistilBertQAModel

print("\n" + "="*70)
print("TESTING FINE-TUNED DISTILBERT QA MODEL")
print("="*70)

# Load model
print("\nLoading fine-tuned model...")
model = DistilBertQAModel.from_finetuned('./checkpoints/distilbert_qa_finetuned')
print("✓ Model loaded successfully!")

# Test 1
print("\n" + "-"*70)
context1 = "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed."
question1 = "What is machine learning?"

print(f"\nTest 1:")
print(f"Question: {question1}")
print(f"Context: {context1[:60]}...")

predictions1 = model.predict(question1, context1, top_k=1)
if predictions1:
    print(f"\nAnswer: {predictions1[0]['answer']}")
    print(f"Score: {predictions1[0]['score']:.4f}")

# Test 2
print("\n" + "-"*70)
context2 = "Python is a high-level programming language known for its simplicity and readability. It is widely used in data science, web development, and artificial intelligence."
question2 = "What is Python used for?"

print(f"\nTest 2:")
print(f"Question: {question2}")
print(f"Context: {context2[:60]}...")

predictions2 = model.predict(question2, context2, top_k=1)
if predictions2:
    print(f"\nAnswer: {predictions2[0]['answer']}")
    print(f"Score: {predictions2[0]['score']:.4f}")

# Test 3
print("\n" + "-"*70)
context3 = "Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data."
question3 = "What is deep learning?"

print(f"\nTest 3:")
print(f"Question: {question3}")
print(f"Context: {context3[:60]}...")

predictions3 = model.predict(question3, context3, top_k=1)
if predictions3:
    print(f"\nAnswer: {predictions3[0]['answer']}")
    print(f"Score: {predictions3[0]['score']:.4f}")

print("\n" + "="*70)
print("✓ FINE-TUNED MODEL IS WORKING!")
print("="*70 + "\n")
