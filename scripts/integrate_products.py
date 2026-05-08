#!/usr/bin/env python3
"""
商品数据整合脚本
将 CatalogBrands、CatalogTypes、CatalogItems 三个文件整合为统一的知识库文档
"""

import json
import os
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def integrate_products():
    source_dir = Path(r"D:\WorkSpace\program\EShop.Core\Catalog.API\Catalog.API\Data\TestData")
    target_dir = Path(r"D:\WorkSpace\program\EShop.Agent\data\documents")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取品牌数据
    with open(source_dir / "CatalogBrands_TestData.txt", "r", encoding="utf-8") as f:
        brands_data = json.load(f)
    brands_map = {item["Id"]: item["Brand"] for item in brands_data}

    # 2. 读取类别数据
    with open(source_dir / "CatalogTypes_TestData.txt", "r", encoding="utf-8") as f:
        types_data = json.load(f)
    types_map = {item["Id"]: item["Type"] for item in types_data}

    # 3. 读取商品数据
    with open(source_dir / "CatalogItems_TestData.txt", "r", encoding="utf-8") as f:
        items_data = json.load(f)

    # 4. 按类别分组
    categories = {}
    for item in items_data:
        brand_name = brands_map.get(item["CatalogBrandId"], "未知品牌")
        type_name = types_map.get(item["CatalogTypeId"], "未知类别")
        type_id = item["CatalogTypeId"]

        product = {
            "name": item["Name"],
            "brand": brand_name,
            "price": item["Price"],
            "description": item["Description"],
            "category": type_name,
            "stock": item["AvailableStock"],
            "restock_threshold": item["RestockThreshold"],
            "max_stock": item["MaxStockThreshold"],
        }

        if type_name not in categories:
            categories[type_name] = {"type_id": type_id, "products": []}
        categories[type_name]["products"].append(product)

    # 5. 生成整合文档
    output_lines = []
    output_lines.append("# 商品目录知识库")
    output_lines.append("")
    output_lines.append(f"本文档整合了商城商品目录的核心数据，包含 {len(items_data)} 款商品，")
    output_lines.append(f"涵盖 {len(categories)} 个品类和 {len(brands_map)} 个品牌。")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")

    # 品类与品牌汇总
    output_lines.append("## 品类概览")
    output_lines.append("")
    for type_name, type_info in categories.items():
        product_count = len(type_info["products"])
        brands_in_category = list(set(p["brand"] for p in type_info["products"]))
        output_lines.append(f"- **{type_name}**：{product_count} 款商品，包含品牌：{', '.join(brands_in_category)}")
    output_lines.append("")

    output_lines.append("## 品牌概览")
    output_lines.append("")
    brand_product_count = {}
    for item in items_data:
        brand_name = brands_map.get(item["CatalogBrandId"], "未知品牌")
        brand_product_count[brand_name] = brand_product_count.get(brand_name, 0) + 1
    for brand in sorted(brand_product_count.keys()):
        count = brand_product_count[brand]
        output_lines.append(f"- **{brand}**：{count} 款商品")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")

    # 各品类商品详情
    for type_name, type_info in categories.items():
        output_lines.append(f"## {type_name}")
        output_lines.append("")
        for i, product in enumerate(type_info["products"], 1):
            output_lines.append(f"### {i}. {product['name']}")
            output_lines.append("")
            output_lines.append(f"- **品牌**：{product['brand']}")
            output_lines.append(f"- **价格**：¥{product['price']:.2f}")
            output_lines.append(f"- **库存**：{product['stock']} 件（补货阈值：{product['restock_threshold']} 件，最大库存：{product['max_stock']} 件）")
            output_lines.append(f"- **描述**：{product['description']}")
            output_lines.append("")

    # 6. 保存整合文档
    output_path = target_dir / "catalog_products_knowledge.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"✅ 整合完成！文档已保存至: {output_path}")
    print(f"📊 统计信息:")
    print(f"   - 总商品数: {len(items_data)}")
    print(f"   - 品类数: {len(categories)}")
    print(f"   - 品牌数: {len(brands_map)}")
    
    # 生成分品类txt文件（便于RAG分块检索）
    for type_name, type_info in categories.items():
        safe_name = type_name.replace("/", "_").replace(" ", "_")
        txt_path = target_dir / f"products_{safe_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# {type_name}\n\n")
            for product in type_info["products"]:
                f.write(f"商品名称：{product['name']}\n")
                f.write(f"品牌：{product['brand']}\n")
                f.write(f"价格：¥{product['price']:.2f}\n")
                f.write(f"库存：{product['stock']} 件\n")
                f.write(f"描述：{product['description']}\n")
                f.write("\n")
        print(f"   - 已生成分品类文件: {txt_path}")

    print("\n🎉 所有文件处理完成！")

if __name__ == "__main__":
    integrate_products()
