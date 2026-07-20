import re

def aggregate_entities(raw_entities: list) -> dict:
    """
    Takes the raw, fragmented output from GLiNER and aggregates it into a clean, unified dictionary structure.
    """
    structured_data = {
        "Name": None,
        "Email": None,
        "Phone": None,
        "Location": None,
        "Experience": [],
        "Education": [],
        "Skills": set()
    }
    
    # Process entities
    for entity in raw_entities:
        label = entity["label"]
        text = entity["text"].strip()
        
        if not text:
            continue
            
        if label == "Name" and not structured_data["Name"]:
            structured_data["Name"] = text
        elif label == "Email" and not structured_data["Email"]:
            # Simple regex validation for email
            if re.match(r"[^@]+@[^@]+\.[^@]+", text):
                structured_data["Email"] = text
        elif label == "Phone" and not structured_data["Phone"]:
            structured_data["Phone"] = text
        elif label == "Location" and not structured_data["Location"]:
            structured_data["Location"] = text
        elif label == "Experience":
            if text not in structured_data["Experience"]:
                structured_data["Experience"].append(text)
        elif label == "Education":
            if text not in structured_data["Education"]:
                structured_data["Education"].append(text)
        elif label == "Skills":
            # Sometimes skills are extracted as comma separated lists: "Python, Java, C++"
            if "," in text:
                sub_skills = [s.strip() for s in text.split(",") if s.strip()]
                structured_data["Skills"].update(sub_skills)
            else:
                structured_data["Skills"].add(text)
                
    # Convert sets back to sorted lists for JSON serialization
    structured_data["Skills"] = sorted(list(structured_data["Skills"]))
    
    return structured_data
