class Model < ApplicationRecord

  has_many :model_color_rels
  has_many :colors, through: :model_color_rels

  has_one :platform_model_rel
  has_one :platform, through: :platform_model_rel

end
