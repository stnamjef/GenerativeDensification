document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
        }


        function objectSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 5
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = './static/images/gobjaverse_1';
                        break;
                    case 1:
                        image.src = './static/images/gobjaverse_2';
                        break;
                    case 2:
                        image.src = './static/images/gso_1';
                        break;
                    case 3:
                        image.src = './static/images/gso_2';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '_gt.jpg';
                        break;
                    case 1:
                        image.src = image.src + '_lara.jpg';
                        break;
                    case 2:
                        image.src = image.src + '_ours.jpg';
                        break;
                    case 3:
                        image.src = image.src + '_coarse.jpg';
                        break;
                    case 4:
                        image.src = image.src + '_fine.jpg';
                        break;

                }
            }

            let scene_list = document.getElementById("object-scale-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function sceneSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics.scene-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = './static/images/re10k_1';
                        break;
                    case 1:
                        image.src = './static/images/re10k_2';
                        break;
                    case 2:
                        image.src = './static/images/re10k_3';
                        break;
                    case 3:
                        image.src = './static/images/re10k_4';
                        break;
                    case 4:
                        image.src = './static/images/re10k_5';
                        break;
                    case 5:
                        image.src = './static/images/re10k_6';
                        break;
                    case 6:
                        image.src = './static/images/re10k_7';
                        break;
                    case 7:
                        image.src = './static/images/re10k_8';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '_gt.jpg';
                        break;
                    case 1:
                        image.src = image.src + '_mvsplat.jpg';
                        break;
                    case 2:
                        image.src = image.src + '_ours.jpg';
                        break;
                }
            }

            let scene_list = document.getElementById("scene-scale-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }