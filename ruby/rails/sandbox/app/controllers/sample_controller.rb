class SampleController < ApplicationController

  attr_reader :query_result

  def index
    # NOP
  end

  def new
    # NOP
  end

  def create
    key1 = params[:data][:key1]
    key2 = params[:data][:key2]
    key1 = "null" if key1.nil?
    key2 = "null" if key2.nil?

    render plain: "PARAS: key1=#{key1}, key2=#{key2}"
  end

  def test_param
    @query_result = []
    @query_result.push("result_0")
    @query_result.push("result_1")
    @query_result.push("result_2")

    render plain: "DONE"
  end

  def query
    ret = ""

    log = "## All Platforms"
    ret << log << "\n"
    logger.debug(log)
    Platform.all.each { |platform|
      ret << "    " << platform.name << "\n"
    }

    ret << "\n"

    log = "## All PlatformModelRels"
    ret << log << "\n"
    logger.debug(log)
    PlatformModelRel.all.each { |platformmodelrel|
      pfid = platformmodelrel.platform_id
      modelid = platformmodelrel.model_id
      platform = Platform.find(pfid)
      model = Model.find(modelid)
      ret << "    " << "#{platform.name} - #{model.name}\n"
    }

    ret << "\n"

    log = "## All Models"
    ret << log << "\n"
    logger.debug(log)
    Model.all.each { |model|
      ret << "    " << model.name << "\n"
    }

    ret << "\n"

    log = "## All ModelColorRels"
    ret << log << "\n"
    logger.debug(log)
    ModelColorRel.all.each { |modelcolorrel|
      modelid = modelcolorrel.model_id
      colorid = modelcolorrel.color_id
      model = Model.find(modelid)
      color = Color.find(colorid)
      ret << "    " << "#{model.name} - #{color.name}\n"
    }

    ret << "\n"

    log = "## All Colors"
    ret << log << "\n"
    logger.debug(log)
    Color.all.each { |color|
      ret << "    " << color.name << "\n"
    }

    ret << "\n"

    log = "## Platform.joins(:models)"
    ret << log << "\n"
    logger.debug(log)
    Platform.joins(:models).each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = "## Platform.joins(:models).select(\"platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name\")"
    ret << log << "\n"
    logger.debug(log)
    Platform.joins(:models)
        .select("platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name")
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = "## Platform.includes(:models).select(\"platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name\")"
    ret << log << "\n"
    logger.debug(log)
    Platform.includes(:models)
        .references(:models)
        .select("platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name")
        .where('models.name = "Rachael"')
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = "## Platform.joins({ :models => :colors}).select(\"platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name\")"
    ret << log << "\n"
    logger.debug(log)
    Platform.joins({ :models => :colors})
        .select("platforms.*, models.*, colors.*, platforms.name AS platform_name, models.name AS model_name, colors.name AS color_name")
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = "## Color.joins({ :models => :platform}).select(\"platforms.*, models.*, coloros.*, platforms.name AS platform_name, models.name AS model_name\")"
    ret << log << "\n"
    logger.debug(log)
    Color.joins({ :models => :platform})
        .select("colors.*, models.*, platforms.*, platforms.name AS platform_name, models.name AS model_name, colors.name AS color_name")
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = '## Color->Model->Platform ALL SELECT'
    ret << log << "\n"
    logger.debug(log)
    Color.joins("LEFT OUTER JOIN model_color_rels ON colors.id = model_color_rels.color_id")
        .joins("LEFT OUTER JOIN models ON model_color_rels.model_id = models.id")
        .joins("LEFT OUTER JOIN platform_model_rels ON models.id = platform_model_rels.model_id")
        .joins("LEFT OUTER JOIN platforms ON platform_model_rels.platform_id = platforms.id")
        .select("colors.*, model_color_rels.*, models.*, platform_model_rels.*, platforms.*, colors.id AS color_id, colors.name AS color_name, models.id AS model_id, models.name AS model_name, platforms.id AS platform_id, platforms.name AS platform_name")
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = '## Color->Model->Platform ALL SELECT'
    ret << log << "\n"
    logger.debug(log)
    Color.joins("LEFT OUTER JOIN model_color_rels ON colors.id = model_color_rels.color_id")
        .joins("LEFT OUTER JOIN models ON model_color_rels.model_id = models.id")
        .joins("LEFT OUTER JOIN platform_model_rels ON models.id = platform_model_rels.model_id")
        .joins("LEFT OUTER JOIN platforms ON platform_model_rels.platform_id = platforms.id")
        .select("colors.*, model_color_rels.*, models.*, platform_model_rels.*, platforms.*, colors.id AS color_id, colors.name AS color_name, models.id AS model_id, models.name AS model_name, platforms.id AS platform_id, platforms.name AS platform_name")
        .where("model_name = ? OR model_name IS NULL", "Rachael")
        .each { |record|
logger.debug("########")
logger.debug("record = #{record.to_s}")
logger.debug("record.variables = #{record.instance_variables}")
      attrs = record.attributes
logger.debug("attrs = #{attrs}")
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = '## Color->Model->Platform ALL SELECT with ActiveRecord'
    ret << log << "\n"
    logger.debug(log)
    Color.eager_load( :models => :platform )
        .select("colors.*, models.*, platforms.*, colors.id AS color_id, colors.name AS color_name, models.id AS model_id, models.name AS model_name, platforms.id AS platform_id, platforms.name AS platform_name")
        .where("model_name = ? OR model_name IS NULL", "Rachael")
        .each { |record|

logger.debug("########")
#logger.debug("record = #{record.to_s}")
#logger.debug("Color.instance_methods(true) = #{Color.instance_methods(true)}")
#logger.debug("record.variables = #{record.instance_variables}")
#logger.debug("record.models.variables = #{record.models.instance_variables}")
#logger.debug("record.models.methods = #{Model.instance_methods(true)}")
logger.debug"## record.models[0] = #{record.models[0]}, len=#{record.models.size}"

          if record.models[0].nil?
              ret << "    color_id=#{record.id}, color_name=#{record.name}, model_id=null, model_name=null, platform_id=null, platform_name=null\n"
          else
            record.models.each { |model|
              ret << "    color_id=#{record.id}, color_name=#{record.name}, model_id=#{model.id}, model_name=#{model.name}, platform_id=#{model.platform.id}, platform_name=#{model.platform.name}\n"
            }
          end
        }

    ret << "\n"



    render plain: ret
  end

end
