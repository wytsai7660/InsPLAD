from typing import final


@final
class Comp:
    name = [
        "glass-insulator",
        "lightning-rod-suspension",
        "polymer-insulator-upper-shackle",
        "vari-grip",
        "yoke-suspension",
    ]
    id = {name: i for i, name in enumerate(name)}

    @staticmethod
    def to_id(name: str):
        return Comp.id[name]

    @staticmethod
    def to_name(id: int):
        return Comp.name[id]


@final
class Stat:
    @staticmethod
    def to_id(name: str):
        return name == "bad"

    @staticmethod
    def to_name(id: bool):
        return "bad" if id else "good"
