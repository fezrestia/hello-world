class Model < ApplicationRecord

  has_many :modelcolorrels
  has_many :colors, through: :modelcolorrels

  has_one :platformmodelrel
  has_one :platform, through: :platformmodelrel

end
