# Simple example for money


class Money:

    def __init__(self, dollars: int, cents: int):
        """
        Cold hard cash, fat stacks, moola, cheddar...
        :param dollars: the big cotton ones
        :param cents: the small metal ones
        """
        self.dollars = dollars
        self.cents = cents

    @property
    def whats_in_the_bank(self):
        return f"{self.dollars} dollars and {self.cents} cents!"


class Person(Money):
    
    def __init__(self, dollars, cents, name: str, nickname: str, favorite_saying: str):
        """
        Defining a person who has some amount of money and a favorite saying.
        :param dollars: inherited from Money
        :param cents: inherited from Money
        :param name: self explanatory
        :param favorite_saying: self explanatory
        """
        super(Person, self).__init__(dollars, cents)
        self.name = name
        self.nickname = nickname
        self.favorite_saying = favorite_saying

    def clean_up_coins(self):
        if self.cents >= 100:
            new_dollars = self.cents // 100
            leftover_cents = self.cents - (new_dollars * 100)
            self.dollars = self.dollars + new_dollars
            self.cents = leftover_cents

    def __str__(self):
        return f'{self.name} aka "{self.nickname}" has {self.whats_in_the_bank}'


donny = Person(dollars=100, cents=20, name='Theodore Donald Kerabatsos', nickname='Donny', favorite_saying='I am the walrus.')
the_dude = Person(dollars=0, cents=69, name='Jeffrey Lebowski', nickname='The Dude', favorite_saying="Obviously, you're not a golfer.")
walter = Person(dollars=400, cents=942, name='Walter Sobchak', nickname='Walter', favorite_saying="Is this your homework Larry?")

