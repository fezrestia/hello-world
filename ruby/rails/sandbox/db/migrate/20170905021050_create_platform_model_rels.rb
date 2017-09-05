class CreatePlatformModelRels < ActiveRecord::Migration[5.1]
  def change
    create_table :platform_model_rels do |t|

      t.integer :platform_id
      t.integer :model_id

      t.timestamps
    end
  end
end
