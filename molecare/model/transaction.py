import datetime as dt
# installing marshmallow as a project dependency
#>pipenv install marshmallow
#Marshmallow is a popular Python package for converting complex datatypes, 
# such as objects, to and from native Python datatypes. 
# Basically, we can use this package to validate, serialize, and deserialize data.
from marshmallow import Schema, fields


class Transaction():
  def __init__(self, description, amount, type):
    self.description = description
    self.amount = amount
    self.created_at = dt.datetime.now()
    self.type = type

  def __repr__(self):
    return '<Transaction(name={self.description!r})>'.format(self=self)


class TransactionSchema(Schema):
  description = fields.Str()
  amount = fields.Number()
  created_at = fields.Date()
  type = fields.Str()

