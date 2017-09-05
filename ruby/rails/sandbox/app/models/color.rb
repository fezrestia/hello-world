class Color < ApplicationRecord

  has_many :modelcolorrels
  has_many :models, through: :modelcolorrels

end
