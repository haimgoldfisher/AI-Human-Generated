from model import predict_generated

if __name__ == '__main__':
    user_input = input("Enter your text: ")
    predict_generated(str(user_input))
