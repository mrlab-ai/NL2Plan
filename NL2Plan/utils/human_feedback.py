def human_feedback(info: str):
    print(info)
    contents = []
    print("Provide feedback (or 'None'). End with ctrl+d.\n")
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    resp = "\n".join(contents)

    if resp.strip().lower() == "none":
        return None # No feedback

    return resp