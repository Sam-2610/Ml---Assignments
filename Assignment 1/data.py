import json

def load_data(filename):
    with open(filename, "r") as f:
        return json.load(f)

def clean_data(data):
    text_to_num = {"one":1,"two":2,"three":3,"four":4,"five":5}

    for user in data["customers"]:
        raw_rating = user.get("rating")

        if isinstance(raw_rating, str):
            raw_rating = raw_rating.strip().lower()
            raw_rating = text_to_num.get(raw_rating, raw_rating)

        user["rating"] = raw_rating

    return data

def handle_missing_values(data):
    for user in data["customers"]:
        if user.get("age") is None:
            user["age"] = None
    return data

def remove_duplications(data):
    seen = set()
    cleaned = []

    for user in data["customers"]:
        user["name"] = user["name"].strip()
        key = (user["name"], user.get("age"))

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(user)

    data["customers"] = cleaned
    return data

def avg_rating(data):
    total, count = 0, 0
    for user in data["customers"]:
        try:
            total += float(user["rating"])
            count += 1
        except:
            pass

    print("Average Rating:", total / count if count else 0)

def poor_rating(data):
    count = 0
    for user in data["customers"]:
        try:
            if float(user["rating"]) < 3:
                count += 1
        except:
            pass

    print("Poor Rating %:", (count / len(data["customers"])) * 100)

def recommendation_feature(data):
    recommendations = []

    for user in data["customers"]:
        rating = float(user.get("rating", 0))

        brand = "Apple" if rating >= 4 else "Samsung"

        recommendations.append({
            "name": user["name"],
            "brand": brand
        })

    return recommendations


# RUN PIPELINE
data = load_data("data.json")

data = clean_data(data)
data = handle_missing_values(data)
data = remove_duplications(data)

avg_rating(data)
poor_rating(data)

print(recommendation_feature(data))