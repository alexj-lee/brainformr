class Car():
    """A class that inherits something.
    """
    def __init__(self, num_of_wheels: int, brand: str):
        """do something

        Parameters
        ----------
        num_of_wheels : int
            wheels
        brand : str
            brand
        """
        self._brand = brand

        
    @property
    def brand(self) -> str:
        """something

        Returns
        -------
        str
            something else
        """
        return self._brand
