class Platform < ApplicationRecord

  has_many :platform_model_rels
  has_many :models, through: :platform_model_rels

end
