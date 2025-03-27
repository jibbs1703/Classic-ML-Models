from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Workclass(Enum):
    PRIVATE = "Private"
    SELF_EMP_NOT_INC = "Self-emp-not-inc"
    LOCAL_GOV = "Local-gov"
    FEDERAL_GOV = "Federal-gov"
    STATE_GOV = "State-gov"
    SELF_EMP_INC = "Self-emp-inc"
    WITHOUT_PAY = "Without-pay"
    NEVER_WORKED = "Never-worked"


class MaritalStatus(Enum):
    NEVER_MARRIED = "Never-married"
    MARRIED_CIV_SPOUSE = "Married-civ-spouse"
    DIVORCED = "Divorced"
    MARRIED_SPOUSE_ABSENT = "Married-spouse-absent"
    SEPARATED = "Separated"
    MARRIED_AF_SPOUSE = "Married-AF-spouse"
    WIDOWED = "Widowed"


class Country(Enum):
    UNITED_STATES = "United-States"
    CUBA = "Cuba"
    JAMAICA = "Jamaica"
    INDIA = "India"
    MEXICO = "Mexico"
    SOUTH = "South"
    PUERTO_RICO = "Puerto-Rico"
    HONDURAS = "Honduras"
    ENGLAND = "England"
    CANADA = "Canada"
    GERMANY = "Germany"
    IRAN = "Iran"
    PHILIPPINES = "Philippines"
    ITALY = "Italy"
    POLAND = "Poland"
    COLUMBIA = "Columbia"
    CAMBODIA = "Cambodia"
    THAILAND = "Thailand"
    ECUADOR = "Ecuador"
    LAOS = "Laos"
    TAIWAN = "Taiwan"
    HAITI = "Haiti"
    PORTUGAL = "Portugal"
    DOMINICAN_REPUBLIC = "Dominican-Republic"
    EL_SALVADOR = "El-Salvador"
    FRANCE = "France"
    GUATEMALA = "Guatemala"
    CHINA = "China"
    JAPAN = "Japan"
    YUGOSLAVIA = "Yugoslavia"
    PERU = "Peru"
    OUTLYING_US = "Outlying-US(Guam-USVI-etc)"
    SCOTLAND = "Scotland"
    TRINIDAD_TOBAGO = "Trinadad&Tobago"
    GREECE = "Greece"
    NICARAGUA = "Nicaragua"
    VIETNAM = "Vietnam"
    HONG = "Hong"
    IRELAND = "Ireland"
    HUNGARY = "Hungary"
    HOLAND_NETHERLANDS = "Holand-Netherlands"
    

class Education(Enum):
    BACHELORS = "Bachelors"
    HS_GRAD = "HS-grad"
    ELEVENTH = "11th"
    MASTERS = "Masters"
    NINTH = "9th"
    SOME_COLLEGE = "Some-college"
    ASSOC_ACDM = "Assoc-acdm"
    ASSOC_VOC = "Assoc-voc"
    SEVENTH_EIGHTH = "7th-8th"
    DOCTORATE = "Doctorate"
    PROF_SCHOOL = "Prof-school"
    FIFTH_SIXTH = "5th-6th"
    TENTH = "10th"
    FIRST_FOURTH = "1st-4th"
    PRESCHOOL = "Preschool"
    TWELFTH = "12th"


class Sex(Enum):
    MALE = "Male"
    FEMALE = "Female"


class Relationship(Enum):
    NOT_IN_FAMILY = "Not-in-family"
    HUSBAND = "Husband"
    WIFE = "Wife"
    OWN_CHILD = "Own-child"
    UNMARRIED = "Unmarried"
    OTHER_RELATIVE = "Other-relative"


class Occupation(Enum):
    TECH_SUPPORT = "Tech-support"
    CRAFT_REPAIR = "Craft-repair"
    OTHER_SERVICE = "Other-service"
    SALES = "Sales"
    EXEC_MANAGERIAL = "Exec-managerial"
    PROF_SPECIALTY = "Prof-specialty"
    HANDLERS_CLEANERS = "Handlers-cleaners"
    MACHINE_OP_INSPECT = "Machine-op-inspct"
    ADM_CLERICAL = "Adm-clerical"
    FARMING_FISHING = "Farming-fishing"
    TRANSPORT_MOVING = "Transport-moving"
    PRIV_HOUSE_SERV = "Priv-house-serv"
    PROTECTIVE_SERV = "Protective-serv"
    ARMED_FORCES = "Armed-Forces"


class Race(Enum):
    WHITE = "White"
    ASIAN_PAC_ISLANDER = "Asian-Pac-Islander"
    AMER_INDIAN_ESKIMO = "Amer-Indian-Eskimo"
    OTHER = "Other"
    BLACK = "Black"


class IncomeInput(BaseModel):
    age: int = Field(..., description="Please input the age of the individual.")
    workclass: str = Field(..., description="Please select the work classification of the individual.")
    fnlwgt: float = Field(..., description="Please provide the final weight, representing the number of people the census believes the entry represents.")
    education: str = Field(..., description="Please input the highest level of education achieved.")
    education_num: int = Field(..., description="Please enter the numerical representation of the education level.")
    marital_status: str = Field(..., description="Please select the marital status of the individual.")
    occupation: str = Field(..., description="Please input the occupation of the individual.")
    relationship: str = Field(..., description="Please specify the relationship status of the individual.")
    race: str = Field(..., description="Please select the race of the individual.")
    sex: str = Field(..., description="Please specify the sex of the individual.")
    capital_gain: float = Field(..., description="Please input the capital gain of the individual.")
    capital_loss: float = Field(..., description="Please input the capital loss of the individual.")
    hours_per_week: int = Field(..., description="Please enter the number of hours worked per week.")
    native_country: str = Field(..., description="Please specify the native country of the individual.")

    @field_validator("*", mode="before")
    @classmethod
    def capitalize_strings(cls, value):
        if isinstance(value, str):
            return value.capitalize()
        return value
    
    class Config:
        json_schema_extra = {
            "example": {
            "age": 39,
            "workclass": "Private",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
        }