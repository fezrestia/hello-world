class Platform < ApplicationRecord

  has_many :platformmodelrels
  has_many :models, through: :platformmodelrels

end
