from dataclasses import dataclass

@dataclass
class DataInstance:
	"""Class for keeping track of a labeled training sample"""
	text: str
	label: str
