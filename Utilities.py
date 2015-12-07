# Auxiliary function - check if a given word represent a number
def check_number(word):
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    found_one_digit = False
    for c in list(word):
        if c in nums:
            found_one_digit = True
    if not found_one_digit:
        return False
    for c in list(word):
        if c != '.' and c != ',' and c not in nums:
            return False
    return True


# Auxiliary function - check if a given word start with a capital, and not the first word of the sentence
def check_capital(word, index):
    if index == 0:
        return False
    return word[0].isupper()


# Auxiliary function - check if the word start\end with a letter, and there's "-" in the middle
def check_bar(word):
    if not word[0].isalpha() or not word[len(word)-1].isalpha():
        return False
    return '-' in word


# Auxiliary function - getting the last n chars
def get_suffix(word, num):
    return word[-num:]


# Auxiliary function - getting the first n chars
def get_prefix(word, num):
    return word[0:num]
