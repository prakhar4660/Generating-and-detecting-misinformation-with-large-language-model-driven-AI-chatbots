from revChatGPT.V1 import Chatbot

chatbot = Chatbot(config={
    "access_token" : "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJwcmFraGFyNDY2MEBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZX0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJ1c2VyX2lkIjoidXNlci1JZDlZRnBzekc3ZlJZTTNGVFJjYUk2M0cifSwiaXNzIjoiaHR0cHM6Ly9hdXRoMC5vcGVuYWkuY29tLyIsInN1YiI6Imdvb2dsZS1vYXV0aDJ8MTA3MTk0ODA4MTY1NTk5ODcyMTcyIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY5MjQzOTE5OSwiZXhwIjoxNjkzNjQ4Nzk5LCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9yZ2FuaXphdGlvbi53cml0ZSBvZmZsaW5lX2FjY2VzcyJ9.rzuHEeVCxzuK2E65BIAijaNU_Ch7mi81iFSx0WGkVhOGFcXPZMQimOZGKSL2RPWQlgJUpUnt9_wBESfPX1mCdDvgwO-Qc6Wwdf__i74L2GLKqBpImlBW1n8vhJoCcMb6ZCOZ3KnY1IM0OatO2cgK8nqUXwW9JAW-OiaXbroF8L-n9ftSzCXBdJ7QvL-khzNy6lMnuedVzoUwtmdTzCKhisEn8KHP6syMn-uu2VQW0f0UBPriqDo_ig6V6Pf-p5Zd3yq33khEIcijrBckIDkmRNTGo6obGwTg1kaMiXlbB0stgKcS4LV-AJT6wUUXh8vxohVdbeIvy5VFv0iJ0zMKMg"
})


def start_chat():
    print('Welcome to ChatGPT CLI')
    while True:
        prompt = input('> ')

        response = ""

        for data in chatbot.ask(
                prompt
        ):
            response = data["message"]

        print(response)


if __name__ == "__main__":
    start_chat()