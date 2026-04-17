import csv
from datetime import datetime
from io import BytesIO, StringIO


def export_to_csv(conversation_history):
    """Export conversation history to CSV"""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Role", "Content"])

    for msg in conversation_history:
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg["role"], msg["content"]])

    return output.getvalue()


def export_to_txt(conversation_history):
    """Export conversation history to plain text"""
    text = "Financial AI Chat History\n"
    text += "=" * 50 + "\n\n"

    for msg in conversation_history:
        role = msg["role"].upper()
        content = msg["content"]
        text += f"{role}:\n{content}\n\n"
        text += "-" * 50 + "\n"

    return text
