class CreateModelColorRels < ActiveRecord::Migration[5.1]
  def change
    create_table :model_color_rels do |t|

      t.integer :model_id
      t.integer :color_id

      t.timestamps
    end
  end
end
