import pandas as pd

eventos = [
    {
        "fecha_aprox": "2015-10",
        "marca_o_tendencia": "Zara",
        "plataforma": "General",
        "tipo_evento": "lanzamiento_sostenibilidad",
        "descripcion_evento": "Zara empieza a dar más visibilidad en España a las prendas con etiqueta 'Join Life' en tiendas físicas y online, introduciendo un discurso de moda más sostenible.",
        "fuente": "Cobertura en medios españoles sobre Zara Join Life (2015-2016)"
    },
    {
        "fecha_aprox": "2016-09",
        "marca_o_tendencia": "Zara",
        "plataforma": "General",
        "tipo_evento": "lanzamiento_sostenibilidad",
        "descripcion_evento": "Se lanza la primera colección destacada 'Join Life' también en la web española de Zara, consolidando la narrativa de sostenibilidad de la marca en el mercado español.",
        "fuente": "Artículos en prensa de moda española sobre Join Life (2016)"
    },
    {
        "fecha_aprox": "2017-02",
        "marca_o_tendencia": "Mango",
        "plataforma": "General",
        "tipo_evento": "lanzamiento_sostenibilidad",
        "descripcion_evento": "Mango presenta en España la colección cápsula 'Mango Committed', con materiales orgánicos y reciclados, posicionándose en el segmento de moda sostenible.",
        "fuente": "Vogue España y otros medios sobre Mango Committed (2017)"
    },
    {
        "fecha_aprox": "2017-04",
        "marca_o_tendencia": "Slow fashion",
        "plataforma": "Instagram",
        "tipo_evento": "activismo_sostenible",
        "descripcion_evento": "En España comienza a ganar fuerza el movimiento Fashion Revolution y el uso de hashtags como #QuienHizoMiRopa y #SlowFashion en Instagram.",
        "fuente": "Fashion Revolution España y medios sobre moda sostenible (2017)"
    },
    {
        "fecha_aprox": "2018-10",
        "marca_o_tendencia": "Moda de segunda mano",
        "plataforma": "Apps / Instagram",
        "tipo_evento": "boom_segunda_mano",
        "descripcion_evento": "Crece el uso en España de apps y perfiles de Instagram dedicados a la venta de ropa de segunda mano y vintage, especialmente entre público joven.",
        "fuente": "Reportajes de prensa española sobre auge de apps de segunda mano (2018-2019)"
    },
    {
        "fecha_aprox": "2019-12",
        "marca_o_tendencia": "Y2K",
        "plataforma": "Instagram+TikTok",
        "tipo_evento": "trend_estetica",
        "descripcion_evento": "La estética Y2K empieza a expandirse en España a través de creadoras de contenido que recuperan prendas y estilos de los años 2000.",
        "fuente": "Artículos en medios de moda españoles sobre retorno Y2K (2019-2020)"
    },
    {
        "fecha_aprox": "2020-03",
        "marca_o_tendencia": "Moda online",
        "plataforma": "General",
        "tipo_evento": "impacto_pandemia",
        "descripcion_evento": "El confinamiento por COVID-19 en España provoca el cierre de tiendas físicas y un aumento repentino de las compras de moda a través de e-commerce.",
        "fuente": "Informes y noticias españolas sobre impacto de la COVID-19 en el comercio minorista (2020)"
    },
    {
        "fecha_aprox": "2020-07",
        "marca_o_tendencia": "Streetwear",
        "plataforma": "TikTok",
        "tipo_evento": "trend_streetwear",
        "descripcion_evento": "Perfiles españoles en TikTok comienzan a popularizar outfits de streetwear y looks urbanos, mezclando marcas de fast fashion con zapatillas deportivas.",
        "fuente": "Contenido de creadores españoles de streetwear en TikTok (2020-2021)"
    },
    {
        "fecha_aprox": "2021-03",
        "marca_o_tendencia": "H&M Conscious",
        "plataforma": "Redes sociales",
        "tipo_evento": "controversia_greenwashing",
        "descripcion_evento": "En España aparecen críticas en redes y medios a la línea H&M Conscious, cuestionando la coherencia entre el marketing verde y los materiales utilizados.",
        "fuente": "Artículos y debates en medios españoles sobre H&M Conscious (2021)"
    },
    {
        "fecha_aprox": "2021-06",
        "marca_o_tendencia": "Shein",
        "plataforma": "TikTok",
        "tipo_evento": "viral_haul",
        "descripcion_evento": "Los 'Shein hauls' realizados por creadoras españolas en TikTok acumulan millones de visualizaciones, mostrando grandes pedidos de ropa barata.",
        "fuente": "Reportajes en medios españoles sobre los hauls de Shein en TikTok (2021)"
    },
    {
        "fecha_aprox": "2021-09",
        "marca_o_tendencia": "Zara",
        "plataforma": "TikTok",
        "tipo_evento": "viral_haul",
        "descripcion_evento": "Se popularizan en España los vídeos de 'Zara haul', donde usuarias enseñan sus compras de nueva temporada y prueban distintas tallas y estilos.",
        "fuente": "Cobertura en medios digitales españoles sobre Zara hauls (2021)"
    },
    {
        "fecha_aprox": "2022-02",
        "marca_o_tendencia": "Choni / trap",
        "plataforma": "TikTok",
        "tipo_evento": "trend_estetica_urbana",
        "descripcion_evento": "En el TikTok español se viralizan contenidos que recuperan la estética 'choni' y estilos vinculados al trap y al reguetón, especialmente entre adolescentes.",
        "fuente": "Artículos y hilos en redes sobre estética choni y trap en España (2022)"
    },
    {
        "fecha_aprox": "2022-03",
        "marca_o_tendencia": "Pija elegante",
        "plataforma": "TikTok+Instagram",
        "tipo_evento": "trend_estetica",
        "descripcion_evento": "La estética 'pija elegante' se populariza en España como adaptación local de la estética old money: americanas, camisas blancas, mocasines y bolsos clásicos.",
        "fuente": "Contenido de creadoras españolas y artículos sobre estética pija elegante (2022)"
    },
    {
        "fecha_aprox": "2022-06",
        "marca_o_tendencia": "Y2K",
        "plataforma": "TikTok",
        "tipo_evento": "trend_estetica",
        "descripcion_evento": "Tiendas y creadoras españolas refuerzan la estética Y2K en redes: tops cortos, pantalones de tiro bajo y accesorios dosmileros aparecen en hauls y looks diarios.",
        "fuente": "Reportajes de moda españoles sobre la consolidación del Y2K (2022)"
    },
    {
        "fecha_aprox": "2022-09",
        "marca_o_tendencia": "Hauls (general)",
        "plataforma": "TikTok+Instagram",
        "tipo_evento": "boom_hauls",
        "descripcion_evento": "En España el formato haul se convierte en contenido habitual: #haul, #zarahaul y #sheinhaul son etiquetas recurrentes en vídeos de moda.",
        "fuente": "Análisis de contenido de moda en TikTok España y medios especializados (2022)"
    },
    {
        "fecha_aprox": "2023-03",
        "marca_o_tendencia": "Zara (Join Life)",
        "plataforma": "General",
        "tipo_evento": "cambio_estrategia_sostenibilidad",
        "descripcion_evento": "Se comunica en medios españoles que Inditex reduce el protagonismo de la etiqueta Join Life y traslada los criterios sostenibles al conjunto de la colección.",
        "fuente": "Noticias económicas y de moda españolas sobre estrategia de sostenibilidad de Inditex (2023)"
    },
    {
        "fecha_aprox": "2023-07",
        "marca_o_tendencia": "Zara",
        "plataforma": "General",
        "tipo_evento": "coleccion_circular",
        "descripcion_evento": "Se presenta en España una colección de Zara con fibras textiles recicladas colaborando con empresas especializadas en reciclaje textil.",
        "fuente": "Vogue España y otros medios sobre colecciones circulares de Zara (2023)"
    },
    {
        "fecha_aprox": "2023-11",
        "marca_o_tendencia": "Shein",
        "plataforma": "TikTok",
        "tipo_evento": "viral_haul",
        "descripcion_evento": "Medios españoles recogen que los hashtags relacionados con Shein hauls en España alcanzan grandes volúmenes de visualizaciones y suscitan debate sobre sobreconsumo.",
        "fuente": "Reportajes en medios españoles sobre Shein y su impacto en el consumo (2023)"
    },
    {
        "fecha_aprox": "2024-04",
        "marca_o_tendencia": "Slow fashion / segunda mano",
        "plataforma": "TikTok+Instagram",
        "tipo_evento": "boom_segunda_mano",
        "descripcion_evento": "Crecen en España los 'thrift hauls' y vídeos de upcycling de prendas en TikTok e Instagram, impulsando la compra de segunda mano entre jóvenes.",
        "fuente": "Artículos en prensa española sobre auge de la segunda mano y la Generación Z (2024)"
    },
    {
        "fecha_aprox": "2025-03",
        "marca_o_tendencia": "Pija elegante / old money",
        "plataforma": "TikTok",
        "tipo_evento": "seguimiento_microtendencias",
        "descripcion_evento": "Trackers de tendencias y medios españoles destacan que estéticas como pija elegante, old money y streetwear conviven en el contenido de moda en TikTok España.",
        "fuente": "Vogue España, prensa de moda y análisis de tendencias en TikTok España (2024-2025)"
    }
]

df = pd.DataFrame(eventos)
output_file = "eventos_moda.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ CSV generado: {output_file}")
print(df.head())
