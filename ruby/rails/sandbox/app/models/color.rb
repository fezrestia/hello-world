class Color < ApplicationRecord

  has_many :model_color_rels
  has_many :models, through: :model_color_rels

end
