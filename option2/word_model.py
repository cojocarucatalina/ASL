import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

words = [
    "mar", "banana", "strugure", "cires", "afina", "capsuna", "mango", "pepene",
    "kiwi", "portocala", "ananas", "pepene verde", "para", "piersica", "caise",
    "pruna", "papaya", "cocos", "lamaie", "lime", "zmeura", "maces", "mandarina",
    "melc", "broasca", "fluture", "pisica", "caine", "cal", "vaca", "frunza",
    "copac", "trandafir", "floare", "iubire", "fericire", "munte", "mare", "lac",
    "rai", "pamant", "stea", "luna", "soare", "cer", "nor", "vint", "ploi", "zapada",
    "frig", "caldura", "vara", "iarna", "primavara", "toamna", "cosmos", "univers",
    "planeta", "galaxie", "stanca", "pietre", "nisip", "plaja", "carte", "pix",
    "creion", "foaie", "masa", "scaun", "fereastra", "usa", "usor", "lumina", "intuneric",
    "strada", "drum", "oras", "sat", "tata", "mama", "frate", "sora", "bunic", "bunicuta",
    "copil", "adolescent", "adult", "batran", "copilasi", "familie", "prieten", "prietena",
    "iubire", "curaj", "bucurie", "frumusete", "grija", "munca", "invatatura", "scoala",
    "student", "profesor", "universitate", "cariera", "bani", "economii", "investitii",
    "afacere", "firma", "comert", "piata", "client", "produse", "servicii", "vanzare",
    "cumparare", "contract", "afaceri", "taxe", "impozit", "reforma", "politica",
    "guvern", "lege", "justitie", "drepturi", "libertate", "egalitate", "democratie",
    "armata", "politie", "spital", "doctor", "nurse", "salut", "multumesc", "pardon",
    "te iubesc", "respect", "pace", "date", "mango", "open", "cat", "dad","manie", "macaroane",
    "cake", "film", "abecedar", "rac", "fizica", "melc", "abatere", "absent", "acceptare", "acoperis",
    "afacere", "afis", "aglomeratie", "ajutor", "alarma", "albina", "alcool", "aliment", "alpinism", "amintire",
    "analiza", "anduranta", "anotimp", "antic", "apa", "apel", "apostrof", "aranjament", "arbitru", "aroma",
    "ascultare", "asigurare", "aspiratie", "atelier", "atmosfera", "autobuz", "aventura", "avion", "baiat",
    "balon", "banca", "baterie", "bautura", "bec", "biblioteca", "bicicleta", "bijuterie", "bluza", "bogatie",
    "bomboana", "bordura", "bucurie", "bufnita", "buton", "cadou", "cafea", "caiet", "calculator", "calitate",
    "campie", "canapea", "capitala", "caracter", "carne", "carte", "castel", "catel", "cerere", "cheie",
    "ciorba", "claritate", "colectie", "comert", "comunicare", "concurenta", "conditie", "conflict", "cont",
    "coridor", "creativitate", "culoare", "curaj", "cuvant", "decor", "delicatete", "demnitate", "detaliu",
    "dezvoltare", "dialog", "diferenta", "dimineata", "domeniu", "dragoste", "educatie", "efort", "echilibru",
    "electricitate", "eleganta", "emotie", "energie", "entuziasm", "erou", "activitate", "actor",
    "adevar", "admiratie", "adunare", "rabbit"
]

# words.sort()
# prefixes = [word[:3] for word in words]
#
# X_train, X_test, y_train, y_test = train_test_split(prefixes, words, test_size=0.2, random_state=42)
#
# model = make_pipeline(CountVectorizer(analyzer='char', ngram_range=(1, 3)), MultinomialNB())
#
# model.fit(X_train, y_train)
#
# with open('predictors.pkl', 'wb') as f:
#     pickle.dump(model, f)
#
# test_word = X_test[0]
# predicted_word = model.predict([test_word])[0]
# print(f"Test prefix: {test_word} -> Predicted word: {predicted_word}")


# Sort words alphabetically
words.sort()

# Create a dictionary to quickly find words by prefix
prefix_dict = {}
for word in words:
    # Store all prefixes from 1 to 3 characters
    for i in range(1, min(4, len(word) + 1)):
        prefix = word[:i]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(word)

# Train the model for normal prediction
prefixes = [word[:3] for word in words]
X_train, X_test, y_train, y_test = train_test_split(prefixes, words, test_size=0.2, random_state=42)
model = make_pipeline(CountVectorizer(analyzer='char', ngram_range=(1, 3)), MultinomialNB())
model.fit(X_train, y_train)


def predict_word(prefix):
    """
    Predicts a word based on a prefix using the following logic:
    1. If the prefix exactly matches a word's beginning, return the alphabetically first such word
    2. If no exact match, try to find words containing the prefix and return the alphabetically first
    3. If still no match, use the ML model to predict
    """
    # Check if we have words starting with this prefix
    if prefix in prefix_dict:
        # Return the alphabetically first word (they're already sorted)
        return prefix_dict[prefix][0]

    # If no exact prefix match, look for words containing this prefix
    matching_words = []
    for word in words:
        if prefix in word:
            matching_words.append(word)

    if matching_words:
        matching_words.sort()
        return matching_words[0]

    # Fall back to the ML model if no matches found
    return model.predict([prefix])[0]


with open('predictors.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'prefix_dict': prefix_dict
    }, f)

test_cases = ['r', 'ra', 'abc', 'z', 'mel']
for test in test_cases:
    predicted = predict_word(test)
    print(f"Prefix: '{test}' → Predicted word: '{predicted}'")

correct = 0
for i, prefix in enumerate(X_test):
    predicted = predict_word(prefix)
    actual = y_test[i]
    if predicted == actual:
        correct += 1
    if i < 5:  # Show just a few examples
        print(f"Test #{i + 1}: Prefix '{prefix}' → Predicted: '{predicted}', Actual: '{actual}'")

accuracy = correct / len(X_test)
print(f"Accuracy on test set: {accuracy:.2%}")
