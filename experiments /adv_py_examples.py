from dataclasses import dataclass
'''
Singleton class in python example. This class is made such that only this particular instance of the class is used everytime.
'''


class A:
    _self = None  # create a flag first to check if any instance is being created or not.

    def __new__(cls):
        if cls._self is None:  # if there is no instance of the class, we create it.
            cls._self = super().__new__(cls)  # the __new__ is used to create an instance of the class
        return cls._self  # return the instance that was created

    def __init__(self):
        self.url = 'https://example.com'

    def track(self):
        print(f"Track event at {self.url}")


# a = A()
# b = A()
# print(a is b) # This returns True since both a and b are the same objects of the class A()

# class method example
'''
Class methods operate on the class itself, not on instances. They're defined with the @classmethod decorator and take cls (the class) as their first parameter.
Class methods are great for:
    Factory methods that create instances in specific ways
    Modifying class variables that apply to all instances
    Alternative constructors
'''


class Shoe:
    size = "large shoe size"  # this is called a class variable

    def __init__(self, company):
        self.company = company

    def description(self):
        return f"A {self.company} shoe with the size of {self.size}"

    @classmethod
    def change_size(cls, shoe_size):
        cls.size = shoe_size
        return f"Shoe size changed to {cls.size}"


'''
Static methods don't operate on the instance or class. They're just regular functions that happen to live in the class namespace. They're defined with the @staticmethod decorator
Static methods are useful when:

    A function is logically related to a class but doesn't need to access class or instance data
    You want to organize utility functions within a class namespace
'''


class MathHelper:
    # static method doesn't access class or instance data
    @staticmethod
    def double(x):
        return x * 2

    @staticmethod
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True


'''
Dataclasses (introduced in Python 3.7) are a way to create classes that are primarily used to store data. They automatically generate special methods like __init__, __repr__, and __eq__.
'''



# Without dataclass - lots of boilerplate
class StudentOld:
    def __init__(self, name, age, grades):
        self.name = name
        self.age = age
        self.grades = grades

    def __repr__(self):
        return f"Student(name='{self.name}', age={self.age}, grades={self.grades})"

    def __eq__(self, other):
        if not isinstance(other, StudentOld):
            return False
        return (self.name == other.name and
                self.age == other.age and
                self.grades == other.grades)


@dataclass
class Student:
    name: str
    age: int
    grades: list

    # you can still add methods
    def average_grade(self):
        return sum(self.grades) / len(self.grades)


if __name__ == '__main__':
    # print(Shoe.size)
    # # this prints "large shoe size"

    # Shoe.change_size(
    #     "Extra small shoe size")  # change the shoe size to small without making an instance of the Shoe class
    # print(Shoe.size)

    # s = Shoe("Goldstar")
    # print(s.description())

    # print(MathHelper.double(5))
    # print(MathHelper.is_prime(5))

    # # Usage
    # alice = Student("Alice", 20, [85, 90, 95])
    # print(alice)  # Output: Student(name='Alice', age=20, grades=[85, 90, 95])
    # print(alice.average_grade())  # Output: 90.0

    # # Equality comparison is automatic
    # bob1 = Student("Bob", 22, [75, 80, 85])
    # bob2 = Student("Bob", 22, [75, 80, 85])
    # print(bob1 == bob2)  # Output: True
    print(Student.name)
