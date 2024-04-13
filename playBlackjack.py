#Blackjack AI using Monte Carlo Simulation and Bayesian Statistics
#Part 1: First generate blackjack strategy using Monte Carlo Simulation. This is done by using bayesian statistics to sample the probability of winning given a certain hand. We make use of python package emcee to run simulations for sampling the probability of winning given a certain hand. We then use the generated strategy to play blackjack and evaluate the performance of the strategy and update the strategy using the results of the game. We then use the updated strategy to play the game again and repeat the process until we have a good strategy for playing blackjack. This process is Monte Carlo.
#Part 2:After the blackjack stratgy is generated, we implement them in a neural network to play blackjack. The use of a neural network is to learn the strategy and improve the strategy over time as the number of cards in the deck reduces which changes the probability of winning given a certain hand over time. The neural network will be trained using the generated strategy from Monte Carlo Simulation and the results of the game will be used to update the strategy. The neural network will be trained using the results of the game and the strategy will be updated using the results of the game. This process is called reinforcement learning.

###Part 1: Blackjack Strategy using Monte Carlo Simulation
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import emcee
import corner
from collections import namedtuple

#Defining the deck of cards, the value of the cards and evaluating the value of the hand
"""
Each card has a value. The value of the card is the number on the card. The value of the face cards is 10. The value of the ace is 11 or 1. The value of the ace is 11 if the total value of the hand is less than or equal to 21. If the total value of the hand is greater than 21, the value of the ace is 1. This value of the hand when the ace is 11 is called soft hand. The value of the hand when the ace is 1 is called hard hand. The value of the hand is the sum of the values of the cards in the hand.
"""
#Defining the deck of cards
def deck():
    deck = []
    for i in range(1, 14):
        for j in range(4):
            if i > 10:
                deck.append(10)
            else:
                deck.append(i)
    return deck

#Defining dealer's shoe
def dealer_shoe():
    shoe = []
    for i in range(6):
        shoe.extend(deck())
    random.shuffle(shoe)
    return shoe

#Define the value of the hand

def value(hand):
    values = []
    for i in hand:
        value = 0
        aces = 0
        for card in i:
            if card == 1:
                aces += 1
                value+=11
            else:
                value += card
        while value > 21 and aces:
            value -= 10
            aces -= 1
        values.append(value)
    return values

### Singular blackjack game

#Definiting the state of the game
State = namedtuple("State", ["player_sum", "dealer_up_card"])

#Rules of the game
"""
Each blackjack game starts with the dealer with only one faced up card and one faced down card, while the player has two faced up cards. The player can see one of the dealer's cards. The player can either hit or stand. If the player hits, the player gets another card. If the player stands, the player does not get another card. The player can hit as many times as the player wants. 
The player wins if the value of the player's hand is greater than the value of the dealer's hand and the value of the player's hand is less than or equal to 21. The player automatically wins if the value of the player's hand is 21 when the player has two cards, this is called blackjack.
The player loses if the value of the player's hand is less than the value of the dealer's hand and the value of the dealer's hand is less than or equal to 21. The player also loses if the dealer has blackjack. The player also loses if the value of the player's hand is greater than 21.
The is a tie if the value of the player's hand is equal to the value of the dealer's hand.
"""

#Defining the rules of the game

#check_victory checks for victory of either the player or the dealer. This function returns 1 if the player wins, -1 if the dealer wins and 0 if there is a tie. This function does not check if the player has bust because the player will lose regardless of what the second card the dealer has or whatever cards the dealer would have drawn. This function also does not check for blackjack because the player will win regardless of what the second card the dealer has or whatever cards the dealer would have drawn. This function only checks for the value of the hand of the player and the dealer and returns the result of the game.
def check_victory(player, dealer_hand,results):
    player_hand_value = player.value
    dealer_hand_value = value(dealer_hand)[0]
    
    for i,ith_player_hand_value in enumerate(player_hand_value):
        if ith_player_hand_value>dealer_hand_value:
            results[i] = 1
        elif ith_player_hand_value<dealer_hand_value:
            results[i] = -1
        else:
            #the case there is a tie
            results[i] = 0
    return results

#Action of the player
"""
The player can have different actions, we define the actions of the player as follows:

1. Hit: The player gets another card.
2. Stand: The player does not get another card.
3. Double: The player doubles the bet and gets another card. The player can only double the bet if the player has two cards. The player cannot get another card after doubling the bet.
4. Split: The player splits the cards into two hands. The player can only split the cards if the player has two cards of the same value. The player can split the cards again if the player has two cards of the same value. The player has access to the other actions after splitting the cards.
5. Surrender: The player surrenders the game and loses half of the bet. The player can only surrender if the player has two cards. (This action is not implemented in this code)
"""

#Defining the hand class and the actions of the player

class playerHand:
    def __init__(self, hands, bets):
        self.hands = hands
        self.value = value(hands)
        self.bets = bets
    
    def hit(self, shoe, hand_index):
        self.hands[hand_index].append(shoe.pop()) #adds card to the hand_index-th hand
        self.value = value(self.hands) #updates the value of the hand
    
    def stand(self,hand_index):
        pass

    def double(self, shoe,hand_index):
        self.hands[hand_index].append(shoe.pop()) #adds card to the hand_index-th hand
        self.value = value(self.hands) #updates the value of the hand
        self.bets[hand_index] *= 2 #doubles the bet for the hand_index-th hand
    
    def split(self, shoe, hand_index): #splits the hand but do not play the game for each hand
        if self.hands[hand_index][0] == self.hands[hand_index][1]:  # Check for identical cards
            original_hand=[self.hands[hand_index][0], shoe.pop()] 
            new_hand = [self.hands[hand_index][1], shoe.pop()]
            self.hands[hand_index] = original_hand
            self.hands.insert(hand_index+1, new_hand)
            self.value = value(self.hands)
            self.bets.insert(hand_index+1, self.bets[hand_index])

        else:
            raise ValueError("Cannot split non-identical cards") 

#Defining the game function
def main():
    SHOE = dealer_shoe()
    #initialise player's bankroll
    #Different from casino blackjack, the player starts with a bankroll of 0, because what we are interested in is the end result and to minimise areas for bugs, the bets are compiled and the player's win/loss is calculated at the end of the game.
    BANKROLL = 0
    #initialise player's bet
    #the bets are initialised like a vector and throughout the game, if there are splits, the bets are appended to the vector. then the results of each hand is compiled as another vector and the resulting win/loss is calculated at the end of the game.
    BETS = [1]
    
    #initialise player and dealer hands
    player_hand = [[SHOE.pop(), SHOE.pop()]]
    dealer_hand = [[SHOE.pop(), SHOE.pop()]]

    #for debugging purposes
    # player_hand = [[10, 10]]
    # dealer_hand = [[5, SHOE.pop()]]

    #initialise player class
    player = playerHand(player_hand, BETS)

    #display player's hand and the dealer's first card, player cannot see both of the dealer's cards
    print("Player hand: ", player_hand)
    print("Dealer hand: ", dealer_hand[0][0])

    #check for blackjack, the game ends if there is a blackjack
    blackjack = check_blackjack(player_hand, dealer_hand)
    if blackjack == 1: #Player blackjack, games end with player win
        win_loss = 1.5*BETS[0] #Blackjack pays 3:2, adjust the ratio according to the casino rules
        BANKROLL += win_loss
        print("Player blackjack")
        print(f"Your gain/loss: {BANKROLL}")
        return int(BANKROLL)
    elif blackjack == -1: #Dealer blackjack, game ends with player loss
        print("Dealer blackjack")
        win_loss = -BETS[0] #For dealer blackjack, the player loses the bet
        BANKROLL += win_loss
        print(f"Your gain/loss: {BANKROLL}")
        return int(BANKROLL)
    
    elif value(player_hand) == 21 and value(dealer_hand) == 21: #Player and dealer blackjack, game ends with a tie
        print("Player and dealer blackjack")
        return int(BANKROLL) #No win/loss for player and dealer blackjack
    
    play_game(player, SHOE, 0)

    #check player bust
    RESULTS = [1 for i in range(len(player.hands))]
    for i,ith_player_hand_value in enumerate(player.value):
        if ith_player_hand_value>21:
            RESULTS[i] = -1

    all_bust = all(element == -1 for element in RESULTS)
    if all_bust:
        BANKROLL = sum(np.array(RESULTS)*np.array(BETS))
        print(f"Your gain/loss: {BANKROLL}")
        return int(BANKROLL)
    
    #dealer draws
    print("Dealer hand: ", dealer_hand)
    print("Dealer draws:")
    dealer_hand = dealer_draws(dealer_hand, SHOE)
    print("Dealer value: ", value(dealer_hand))

    #check dealer bust
    if value(dealer_hand)[0] > 21:
        print("Dealer bust")
        BANKROLL = sum(np.array(RESULTS)*np.array(BETS))
        print(f"Your gain/loss: {BANKROLL}")
        return int(BANKROLL)

    #check for victory
    RESULTS = check_victory(player, dealer_hand, RESULTS)
    BANKROLL = sum(np.array(RESULTS)*np.array(BETS))

    print(f"Your gain/loss: {BANKROLL}")
    return int(BANKROLL)

#check for blackjack
def check_blackjack(player_hand, dealer_hand):
    player_value = value(player_hand)[0]
    dealer_value = value(dealer_hand)[0]
    #check if player has blackjack
    if player_value == 21 and dealer_value != 21 :
        blackjack = 1
        return blackjack
    #check if dealer has a blackjack
    elif dealer_value == 21 and player_value != 21:
        blackjack = -1
        return blackjack
    else:
        blackjack = 0
        return blackjack


#dealer plays
def dealer_draws(dealer_hand, shoe):
    dealer_value = value(dealer_hand)[0]
    while dealer_value < 17:
        dealer_hand[0].append(shoe.pop())
        dealer_value = value(dealer_hand)[0]
        print("Dealer hand: ", dealer_hand)
    return dealer_hand

#check for bust

def check_bust(hand):
    if sum(hand)>21:
        return True
    else:
        return False

#Defining the game function
def play_game(player, shoe, hand_index):
    
    
    #initialise move
    move = ''
    while move != 'stand' and player.value[hand_index] <= 21:
        if player.hands[hand_index][0] == player.hands[hand_index][1]:
            can_split = True
        else:
            can_split = False

        if len(player.hands[hand_index])==2:
            can_double = True
        else:
            can_double = False

        move = input("Enter your move: ").lower()
        
        if move == 'stand':
            return
        
        elif move == 'hit':
            player.hit(shoe, hand_index)
            print("Player hand: ", player.hands)

        elif move == 'double' and can_double:
            player.double(shoe, hand_index)
            print("Player hand: ", player.hands)
            return
        
        elif move == 'double' and not can_double:
            print("Cannot double with more than 2 cards")

        elif move == 'split' and can_split:
            player.split(shoe, hand_index)
            print("Player hand: ", player.hands)
            play_game(player, shoe, hand_index)
            play_game(player, shoe, hand_index+1)
            return
        elif move == 'split' and not can_split:
            print("Cannot split non-identical cards")
        
        else:
            print("Invalid move")
            
        # print("Player hand: ", player.hands)
        if player.value[hand_index]>21:
            print("Player bust")
            return


#start the game
play = True
bankroll = int(input("Enter your bankroll: "))
while play:
    win_loss=main()
    bankroll += win_loss
    print("Your current bankroll: ", bankroll)
    play = input("Do you want to play again? (yes/no): ").lower()
    if play == 'no':
        break
print("Game over")
print(f"Your final bankroll: {bankroll}")

#     hand_bet = bet
#     if move == 'double':
#         bankroll -= hand_bet
#         hand_bet *= 2
#         player_hand.append(shoe.pop())
#         #check for player bust
#         if check_bust(player_hand) is True:
#             print("Player bust")
#             return
        
        
    
#     elif move == 'split':
#         player_hand_1 = [player_hand[0], shoe.pop()]
#         player_hand_2 = [player_hand[1], shoe.pop()]
#         bankroll -= bet
#         play_game(player_hand_1, dealer_hand, shoe, bet, bankroll, move)
#         play_game(player_hand_2, dealer_hand, shoe, bet, bankroll, move)

#         return bankroll
#     while move != 'stand' or value(player_hand)>21: #the game ends when the player stands or if the player busts
#         move = input("Enter your move: ")
#         if move == 'hit':
#             player_hand.append(shoe.pop())
        
#         elif move == 'double':
#             player_hand.append(shoe.pop())
#             bankroll -= bet
#             bet *= 2
            
            
#         elif move == 'split':
#             player_hand.split(shoe)
#             move = 'stand'
#         else:
#             raise ValueError("Invalid move")
#         move = input("Enter your move: ")

#         #check if player has bust
#         if value(player_hand)>21:
#             print("Player bust")
    

#     while dealer_hand.value < 17:
#         dealer_hand.hit(shoe)

#     #check if player has blackjack
#     if player_hand.value == 21 and len(player_hand.hand) == 2 and dealer_hand.value != 21 and len(dealer_hand.hand) == 2:
#         win_loss = 1
#         return win_loss
#     #check if dealer has a blackjack
#     elif dealer_hand.value == 21 and len(dealer_hand.hand) == 2 and player_hand.value != 21 and len(dealer_hand.hand) == 2:
#         win_loss = -1
#         return win_loss
#     else:
#         win_loss=check_victory(player_hand.hand, dealer_hand.hand)
#         return win_loss
# #dealer draws
#     dealer_hand = dealer_draws(dealer_hand, shoe)

#     #check dealer bust
#     if check_bust(dealer_hand) is True:
#         print("Dealer bust")
#         win_loss = 2*bet
#         bankroll += win_loss
#         return

#     #check for victory
#     victory = check_victory(player_hand, dealer_hand)
#     if victory == 1:
#         win_loss = 2*bet
#         bankroll += win_loss
#         return
#     elif victory == -1:
#         return
#     else:
#         win_loss = bet
#         bankroll += win_loss
#         return


# #Defining the hand class and the actions of the player

# class hand:
#     def __init__(self, hand):
#         self.hand = hand
#         self.value = value(hand)
#     def hit(self, shoe):
#         self.hand.append(shoe.pop())
#         self.value = value(self.hand)
#     def stand(self):
#         pass
#     def double(self, shoe):
#         self.hand.append(shoe.pop())
#         self.value = value(self.hand)
#     def split(self, shoe):
#         if self.hand[0] == self.hand[1]:  # Check for identical cards
#             self.hand = [[self.hand[0], shoe.pop()], [self.hand[1], shoe.pop()]]
#             self.value = value(self.hand)
#         else:
#             raise ValueError("Cannot split non-identical cards") 

# #Defining the action of the player
# Action = namedtuple("Action", ["hit", "stand", "double", "split"])