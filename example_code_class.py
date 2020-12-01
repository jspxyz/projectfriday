# from models import *
# model_a_conv2d() # must write the function
# test_b()

# the  class does not contain any data
# until the object is defined outside of the class
# the created object has the behavior of the class
class Human:
    def __init__(self):
        self.no_of_legs = 2

class Student(Human): 
    def __init__(self, first_name, last_name, age):
        super().__init__()
        self.name = first_name + last_name
        self.age = age

    # this is an instant method
    # no decorator (static or class)
    def greeting(self):
        print("Hello, " + self.name)

    def select_class(self, class_name):
        self.class_ = class_name

    @staticmethod
    def say_goodbye():
        print("Good Bye")

    # staticmethod vs. this method
    # def say_goodbye(self)
    #     print("Good Bye")

    # the class is the blueprint
    # each object made with the class is the house
    @classmethod
    def add_James(cls):
        return cls("James", "Park", 30)

# this is the object (aka the house in this example)
jp = Student("James", "Park", 30)

print(jp.name)
jp.greeting()
jp.select_class("Kermadec")
print(jp.class_)
# jp.say_goodbye()
print(Student.add_James().no_of_legs)
